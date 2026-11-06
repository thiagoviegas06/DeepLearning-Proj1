"""CIFAR-style ResNet for the Kaggle CIFAR-10 ResNet architecture challenge (<= 5M parameters).

Depth, width and kernel size are flags; the default is the 5x5-conv ResNet-20 baseline (3.0M
params). Data (data/): 50,000 labeled 32x32 RGB training images (5,000 per class) and 2,000
unlabeled Kaggle test images. The labeled set is split per class into train / val / test (default
40,000 / 5,000 / 5,000). Val picks the best epoch; the local test split is scored once at the end.
With --val-frac 0 --test-frac 0 the model trains on all 50,000 and the last epoch is kept.
The PNGs are decoded once into data/cache.npz, and the whole dataset lives on the device. All
augmentation (color jitter, random grayscale, sharpness, RandAugment, flip, pad-and-crop, zoom, Cutout,
Mixup / CutMix) runs on the GPU per batch. The local test split and the Kaggle images are scored both plain and with flip TTA.

Knowledge distillation: train a larger teacher first (it may exceed the 5M limit with
--teacher-over-limit; it is never submitted), then train the student with --teacher TEACHER_RUN/best.pt.
The teacher sees the same augmented batches and the student matches its softened predictions. Several
teachers (--teacher A/best.pt B/best.pt ...) act as an ensemble: the student matches their averaged
logits. Teachers that should differ only in their weights share the student's --seed (the data split)
and get their own --train-seed.

Sharpness-aware minimization (--sam RHO, Foret et al. 2021): each step first climbs to the worst
weights within an L2 ball of radius RHO (one extra forward/backward), and applies the gradient taken
there to the original weights, which steers training to flat minima. Doubles the time per epoch.

  python train.py                                   # 5x5 ResNet-20 baseline, 100 epochs, early stopping
  python train.py --smoke                           # CPU check, tiny subset
  python train.py --blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 \
                  --label-smoothing 0.1 --cutout 16 --mix cutmix     # 4.7M-param recipe
  python train.py ... --width 128 --teacher-over-limit --out runs/teacher   # 18.8M-param teacher
  python train.py ... --width 128 --teacher-over-limit --train-seed 1 --out runs/teacher_s1   # 2nd teacher, same split
  python train.py ... --width 64 --teacher runs/teacher/best.pt runs/teacher_s1/best.pt      # student of the ensemble

Writes OUT/history.json, summary.json, curves.png, best.pt, submission.csv and submission_tta.csv (id,label).
Run `python model_summary.py --run OUT` afterwards for the layer table, parameter count and test accuracy.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PARAM_LIMIT = 5_000_000  # competition rule: at most 5M trainable parameters


def load_images(data_dir, cache):
    """Return (x_train uint8 NHWC, y_train, x_kaggle uint8 NHWC, kaggle_ids), decoding the PNGs once."""
    if cache.exists():
        z = np.load(cache)
        return z["x_train"], z["y_train"], z["x_kaggle"], z["kaggle_ids"]
    from PIL import Image

    def read(folder, names):
        return np.stack([np.asarray(Image.open(folder / n).convert("RGB")) for n in names])

    with open(data_dir / "train_labels.csv") as f:
        train = sorted(csv.DictReader(f), key=lambda r: int(r["id"]))
    with open(data_dir / "test.csv") as f:
        test = sorted(csv.DictReader(f), key=lambda r: int(r["id"]))
    x_train = read(data_dir / "train_images", [r["filename"] for r in train])
    y_train = np.array([int(r["label"]) for r in train], dtype=np.int64)
    x_kaggle = read(data_dir / "test_images", [r["filename"] for r in test])
    kaggle_ids = np.array([int(r["id"]) for r in test], dtype=np.int64)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, x_train=x_train, y_train=y_train, x_kaggle=x_kaggle, kaggle_ids=kaggle_ids)
    tmp.replace(cache)
    return x_train, y_train, x_kaggle, kaggle_ids


def stratified_split(y, val_frac, test_frac, seed):
    """Per-class shuffle, then carve val and test off the front of each class."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_val, n_test = round(val_frac * len(idx)), round(test_frac * len(idx))
        val.append(idx[:n_val])
        test.append(idx[n_val:n_val + n_test])
        train.append(idx[n_val + n_test:])
    return [np.sort(np.concatenate(s)) for s in (train, val, test)]


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, kernel, stride):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(cin, cout, kernel, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, kernel, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        if stride == 1 and cin == cout:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """CIFAR-style ResNet: stem conv, one stage per entry of `blocks` (width doubles and
    resolution halves after the first), global average pool, linear. Depth = 2*sum(blocks) + 2."""

    def __init__(self, blocks=(3, 3, 3), width=32, kernel=5, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, kernel, 1, kernel // 2, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU())
        layers, cin = [], width
        for i, n in enumerate(blocks):
            cout = width * 2 ** i
            for j in range(n):
                layers.append(BasicBlock(cin, cout, kernel, 2 if (i > 0 and j == 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def color_jitter(x, brightness, contrast, saturation):
    """Per-image random brightness, contrast and saturation on [0, 1] pixels, like torchvision's
    ColorJitter: each factor is drawn from U(1 - s, 1 + s), applied in that order, then clamped."""
    n, dev = x.shape[0], x.device
    factor = lambda s: 1 + s * (2 * torch.rand(n, 1, 1, 1, device=dev) - 1)
    gray = lambda t: (t * torch.tensor([0.299, 0.587, 0.114], device=dev)[:, None, None]).sum(1, keepdim=True)
    if brightness:
        x = x * factor(brightness)
    if contrast:
        m = gray(x).mean((2, 3), keepdim=True)
        x = (x - m) * factor(contrast) + m
    if saturation:
        g = gray(x)
        x = (x - g) * factor(saturation) + g
    return x.clamp(0, 1)


def random_grayscale(x, prob):
    """Like torchvision's RandomGrayscale on [0, 1] pixels: with probability `prob` per image, replace the
    three channels by their luminance."""
    g = (x * torch.tensor([0.299, 0.587, 0.114], device=x.device)[:, None, None]).sum(1, keepdim=True)
    apply = torch.rand(x.shape[0], device=x.device) < prob
    return torch.where(apply[:, None, None, None], g.expand_as(x), x)


def sharpen(x, factor, prob):
    """Like torchvision's RandomAdjustSharpness on [0, 1] pixels: with probability `prob` per image,
    blend it with a 3x3-smoothed copy, out = blur + factor * (x - blur), so factor > 1 sharpens.
    Border pixels are left unsmoothed, as in torchvision."""
    k = torch.tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]], device=x.device) / 13
    blur = x.clone()
    blur[:, :, 1:-1, 1:-1] = F.conv2d(x, k.expand(x.shape[1], 1, 3, 3), groups=x.shape[1])
    out = (blur + factor * (x - blur)).clamp(0, 1)
    apply = torch.rand(x.shape[0], device=x.device) < prob
    return torch.where(apply[:, None, None, None], out, x)


def equalize(x):
    """Per-image, per-channel histogram equalization of [0, 1] pixels, as in torchvision's equalize."""
    v = (x * 255).round().long().flatten(2)
    hist = torch.zeros(*v.shape[:2], 256, dtype=torch.long, device=x.device).scatter_add_(2, v, torch.ones_like(v))
    last = hist.gather(2, ((hist > 0) * torch.arange(256, device=x.device)).argmax(2, keepdim=True))
    step = (hist.sum(2, keepdim=True) - last) // 255
    lut = (hist.cumsum(2) + step // 2) // step.clamp(min=1)
    lut = torch.cat([torch.zeros_like(lut[..., :1]), lut[..., :-1]], 2).clamp(0, 255)
    out = lut.gather(2, v).view_as(x).float() / 255
    return torch.where(step[..., None] == 0, x, out)


def rand_augment(x, n_ops, magnitude, bins=31):
    """RandAugment (Cubuk et al., 2020) on [0, 1] pixels with torchvision's 14 ops and magnitude scales:
    `n_ops` rounds, each applying an op drawn uniformly and independently per image, at `magnitude`
    out of bins - 1 (9 of 30 is the usual CIFAR setting), with a random sign for the signed ops.
    Geometric ops fill uncovered pixels with black, like torchvision."""
    n, _, h, w = x.shape
    dev = x.device
    m = magnitude / (bins - 1)
    gray = lambda t: (t * torch.tensor([0.299, 0.587, 0.114], device=dev)[:, None, None]).sum(1, keepdim=True)
    blend = lambda a, b, f: (b + f * (a - b)).clamp(0, 1)  # f = 1 keeps a, f = 0 gives b
    for _ in range(n_ops):
        op = torch.randint(0, 14, (n,), device=dev)
        sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        # 0-4 geometric: shear x, shear y, translate x, translate y, rotate, as one affine resample
        theta = torch.eye(2, 3, device=dev).repeat(n, 1, 1)
        shear, shift, angle = 0.3 * m * sign, 2 * (150 / 331) * m * sign, math.radians(30 * m) * sign
        theta[:, 0, 1] = torch.where(op == 0, shear, theta[:, 0, 1])
        theta[:, 1, 0] = torch.where(op == 1, shear, theta[:, 1, 0])
        theta[:, 0, 2] = torch.where(op == 2, shift, theta[:, 0, 2])
        theta[:, 1, 2] = torch.where(op == 3, shift, theta[:, 1, 2])
        rot = op == 4
        theta[rot, :, :2] = torch.stack([torch.stack([angle.cos(), -angle.sin()], 1),
                                         torch.stack([angle.sin(), angle.cos()], 1)], 1)[rot]
        geo = op <= 4
        if geo.any():
            grid = F.affine_grid(theta[geo], [int(geo.sum()), *x.shape[1:]], align_corners=False)
            x[geo] = F.grid_sample(x[geo], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # 5-8 brightness, color, contrast, sharpness: factor 1 +- 0.9 m
        f = (1 + 0.9 * m * sign)[:, None, None, None]
        color_ops = {5: lambda t, f: blend(t, torch.zeros_like(t), f),
                     6: lambda t, f: blend(t, gray(t), f),
                     7: lambda t, f: blend(t, gray(t).mean((2, 3), keepdim=True), f),
                     8: lambda t, f: sharpen(t, f, 1.0)}
        for k, fn in color_ops.items():
            sel = op == k
            if sel.any():
                x[sel] = fn(x[sel], f[sel])
        # 9 posterize, 10 solarize, 11 autocontrast, 12 equalize, 13 identity
        bits = 8 - int(round(m * 4))
        level = 2 ** (8 - bits)
        lo, hi = x.amin((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)
        other_ops = {9: lambda t, lo, hi: torch.floor((t * 255).round() / level) * level / 255,
                     10: lambda t, lo, hi: torch.where(t >= 1 - m, 1 - t, t),
                     11: lambda t, lo, hi: torch.where(hi > lo, (t - lo) / (hi - lo).clamp(min=1e-8), t),
                     12: lambda t, lo, hi: equalize(t)}
        for k, fn in other_ops.items():
            sel = op == k
            if sel.any():
                x[sel] = fn(x[sel], lo[sel], hi[sel])
    return x


def augment(x, pad=4):
    """Random horizontal flip + random crop with zero padding (zero = channel mean after normalizing)."""
    n, c, h, w = x.shape
    dev = x.device
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = F.pad(x, (pad, pad, pad, pad))
    rows = torch.randint(0, 2 * pad + 1, (n, 1), device=dev) + torch.arange(h, device=dev)
    cols = torch.randint(0, 2 * pad + 1, (n, 1), device=dev) + torch.arange(w, device=dev)
    return x[torch.arange(n, device=dev)[:, None, None, None], torch.arange(c, device=dev)[None, :, None, None],
             rows[:, None, :, None], cols[:, None, None, :]]


def zoom(x, lo, hi):
    """Random scale per image, factor ~ U(lo, hi): > 1 enlarges the object (the border is cropped away),
    < 1 shrinks it and zero-fills the margin (zero = channel mean after normalization). Bilinear
    resampling about the image center through an affine grid. Motivated by the Kaggle images, whose
    objects are framed at more varied scales than CIFAR's (close-ups, small objects in a wide scene)."""
    n = x.shape[0]
    s = lo + (hi - lo) * torch.rand(n, device=x.device)
    theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / s
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def cutout(x, size):
    """Cutout (DeVries & Taylor, 2017): zero one size x size square per image, centered at a uniform
    random pixel so it may be clipped by the border. Applied after normalization, so zero is the
    per-channel mean."""
    n, _, h, w = x.shape
    dev = x.device
    cy = torch.randint(0, h, (n, 1, 1), device=dev)
    cx = torch.randint(0, w, (n, 1, 1), device=dev)
    ys = torch.arange(h, device=dev)[None, :, None]
    xs = torch.arange(w, device=dev)[None, None, :]
    hole = (ys >= cy - size // 2) & (ys < cy + size // 2) & (xs >= cx - size // 2) & (xs < cx + size // 2)
    return x.masked_fill(hole[:, None], 0.0)


def mix_batch(x, y, mode, alpha):
    """Mixup (Zhang et al., 2018) or CutMix (Yun et al., 2019) with one lambda ~ Beta(alpha, alpha)
    per batch, pairing each image with a shuffled partner. Returns (x_mixed, y_a, y_b, lam); the
    loss is lam * CE(y_a) + (1 - lam) * CE(y_b)."""
    n, _, h, w = x.shape
    perm = torch.randperm(n, device=x.device)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    if mode == "mixup":
        return lam * x + (1 - lam) * x[perm], y, y[perm], lam
    # cutmix: paste a random box of the partner image whose area is (1 - lam) of the image,
    # then set lam to the exact unpasted fraction (the box can be clipped by the border).
    rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
    cy, cx = torch.randint(0, h, (1,)).item(), torch.randint(0, w, (1,)).item()
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    return x, y, y[perm], 1 - (y2 - y1) * (x2 - x1) / (h * w)


@torch.no_grad()
def logits_of(model, x, batch, amp, tta=False):
    """Logits for x in batches; with tta, the average of the logits of each image and its mirror."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            z = model(xb).float()
            if tta:
                z = (z + model(xb.flip(3)).float()) / 2
        out.append(z)
    return torch.cat(out)


def evaluate(model, x, y, batch, amp, tta=False):
    """(mean cross-entropy, accuracy); (nan, nan) for an empty split."""
    if len(x) == 0:
        return float("nan"), float("nan")
    logits = logits_of(model, x, batch, amp, tta)
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def predict(model, x, batch, amp, tta=False):
    return logits_of(model, x, batch, amp, tta).argmax(1)


def write_submission(path, ids, preds):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(zip(ids.tolist(), preds.tolist()))


def load_run(path, dev):
    """(model in eval mode, checkpoint dict) from a best.pt written by this script."""
    ckpt = torch.load(path, map_location=dev)
    a = ckpt["args"]
    model = ResNet(tuple(a["blocks"]), a["width"], a["kernel"]).to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


def sam_perturb(params, rho):
    """Move each parameter to w + rho * g / ||g|| (g = current grads, one global norm) and return the offsets."""
    grads = [q.grad for q in params if q.grad is not None]
    scale = rho / (torch.norm(torch.stack([g.norm(2) for g in grads]), 2) + 1e-12)
    offsets = []
    with torch.no_grad():
        for q in params:
            e = q.grad * scale if q.grad is not None else None
            if e is not None:
                q.add_(e)
            offsets.append(e)
    return offsets


def sam_restore(params, offsets):
    with torch.no_grad():
        for q, e in zip(params, offsets):
            if e is not None:
                q.sub_(e)


def set_bn_momentum(model, value):
    """Set every BatchNorm's momentum (0 freezes the running stats); returns the previous values."""
    prev = []
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            prev.append((m, m.momentum))
            m.momentum = value
    return prev


def kd_loss(student_logits, teacher_logits, temp):
    """Hinton et al. (2015) distillation loss: KL(teacher || student) on temperature-softened
    distributions, scaled by temp^2 so its gradients stay comparable to the cross-entropy's."""
    return F.kl_div(F.log_softmax(student_logits / temp, 1), F.log_softmax(teacher_logits / temp, 1),
                    reduction="batchmean", log_target=True) * temp ** 2


def plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="val")
    a1.set(xlabel="epoch", ylabel="cross-entropy", title="loss")
    a2.plot(ep, [h["train_acc"] for h in history], label="train (augmented)")
    a2.plot(ep, [h["val_acc"] for h in history], label="val")
    a2.set(xlabel="epoch", ylabel="accuracy", title="accuracy")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=HERE / "data")
    p.add_argument("--out", type=Path, default=HERE / "runs" / "baseline")
    p.add_argument("--blocks", type=int, nargs="+", default=[3, 3, 3], help="basic blocks per stage")
    p.add_argument("--width", type=int, default=32, help="channels in the first stage")
    p.add_argument("--kernel", type=int, default=5, help="conv kernel size in the stem and every block")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20,
                   help="stop after this many epochs without a new best val accuracy (0 = off)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup", type=float, default=1.0, help="linear warmup, in epochs")
    p.add_argument("--jitter", type=float, nargs=3, default=[0.2, 0.2, 0.2], metavar=("B", "C", "S"),
                   help="color jitter strength for brightness, contrast, saturation (0 0 0 = off)")
    p.add_argument("--grayscale", type=float, default=0.0,
                   help="probability that a training image is converted to grayscale (0 = off)")
    p.add_argument("--sharpness", type=float, default=0.0,
                   help="sharpness factor for random sharpening, e.g. 2.0 (0 = off)")
    p.add_argument("--sharpness-prob", type=float, default=0.5, help="probability that an image is sharpened")
    p.add_argument("--randaugment", type=int, nargs=2, default=[0, 0], metavar=("N", "M"),
                   help="RandAugment: N ops per image at magnitude M of 30 (e.g. 2 9; 0 0 = off)")
    p.add_argument("--zoom", type=float, nargs=2, default=[1.0, 1.0], metavar=("LO", "HI"),
                   help="random scale factor per training image, uniform in [LO, HI] (e.g. 0.7 1.4; 1 1 = off)")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="cross-entropy label smoothing (0 = off)")
    p.add_argument("--cutout", type=int, default=0,
                   help="side of the square hole Cutout zeroes in each training image (0 = off; 16 is standard for CIFAR)")
    p.add_argument("--mix", choices=["none", "mixup", "cutmix"], default="none", help="batch mixing augmentation")
    p.add_argument("--mix-alpha", type=float, default=1.0, help="mixing weight lambda ~ Beta(alpha, alpha)")
    p.add_argument("--mix-prob", type=float, default=1.0, help="probability that a batch is mixed")
    p.add_argument("--teacher", type=Path, nargs="+",
                   help="best.pt of one or more trained teacher runs to distill from; several are averaged (ensemble)")
    p.add_argument("--kd-alpha", type=float, default=0.9,
                   help="weight of the distillation loss; the label loss gets 1 - alpha")
    p.add_argument("--kd-temp", type=float, default=4.0, help="distillation softmax temperature")
    p.add_argument("--sam", type=float, default=0.0,
                   help="sharpness-aware minimization with neighbourhood radius RHO (0 = off; 0.05 is the CIFAR value)")
    p.add_argument("--teacher-over-limit", action="store_true",
                   help="allow more than 5M params; only for training a distillation teacher, never for submission")
    p.add_argument("--val-frac", type=float, default=0.1, help="0 = no val split; the last epoch is kept")
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="seeds the split and, unless --train-seed is set, training")
    p.add_argument("--train-seed", type=int, default=None,
                   help="seed for the weights and the training randomness; defaults to --seed. Use it to train "
                        "several teachers on the same split")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="1 epoch on 1,000 train images, tiny eval sets")
    args = p.parse_args()
    if args.smoke:
        args.epochs = 1
    has_val = args.val_frac > 0
    if args.train_seed is None:
        args.train_seed = args.seed
    as_json = lambda v: [str(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], Path) else str(v) if isinstance(v, Path) else v
    saved_args = {k: as_json(v) for k, v in vars(args).items()}  # for best.pt and summary.json

    t0 = time.time()
    x_all, y_all, x_kaggle, kaggle_ids = load_images(args.data, args.data / "cache.npz")
    tr, va, te = stratified_split(y_all, args.val_frac, args.test_frac, args.seed)
    if args.smoke:
        tr, va, te = tr[:1000], va[:200], te[:200]
    print(f"data: train {len(tr)}  val {len(va)}  test {len(te)}  kaggle {len(x_kaggle)}  "
          f"(loaded in {time.time() - t0:.1f}s)")

    dev = torch.device(args.device)
    amp = dev.type == "cuda"
    to_float = lambda a: torch.from_numpy(a).to(dev).permute(0, 3, 1, 2).float().div_(255)
    x_train = to_float(x_all[tr])
    mean = x_train.mean((0, 2, 3), keepdim=True)
    std = x_train.std((0, 2, 3), keepdim=True)
    norm = lambda x: (x - mean) / std  # x_train stays in [0, 1] so color jitter can run before normalizing
    x_val, x_test, x_kag = (norm(to_float(a)) for a in (x_all[va], x_all[te], x_kaggle))
    y_train, y_val, y_test = (torch.from_numpy(y_all[i]).to(dev) for i in (tr, va, te))

    torch.manual_seed(args.train_seed)
    model = ResNet(tuple(args.blocks), args.width, args.kernel).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(q.numel() for q in model.parameters())
    depth = 2 * sum(args.blocks) + 2
    print(f"model: ResNet-{depth}, {args.kernel}x{args.kernel} convs, blocks {args.blocks}, "
          f"width {args.width}, {n_params:,} params, device {dev}")
    if n_params > PARAM_LIMIT and not args.teacher_over_limit:
        raise SystemExit(f"{n_params:,} parameters exceeds the competition limit of {PARAM_LIMIT:,}")
    if n_params > PARAM_LIMIT:
        print(f"TEACHER RUN: {n_params:,} params, over the {PARAM_LIMIT:,} limit; do not submit its predictions")
    criterion = lambda logits, y: F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)

    teachers = []  # (model, to_teacher) pairs; the student matches their averaged logits
    if args.teacher:
        for path in args.teacher:
            # summary.json is written only when a run finishes, so this refuses a teacher that is still training
            t_summary = path.parent / "summary.json"
            if not t_summary.exists():
                raise SystemExit(f"{t_summary} not found: the teacher run has not finished")
            with torch.random.fork_rng(devices=[]):  # building the teacher must not shift the training RNG stream
                teacher, t_ckpt = load_run(path, dev)
            if t_ckpt["epoch"] != json.loads(t_summary.read_text())["best_epoch"]:
                raise SystemExit(f"{path} is not the final best checkpoint in {t_summary}")
            for q in teacher.parameters():
                q.requires_grad_(False)
            # the student's batches are normalized with the student's stats; re-normalize for the teacher
            t_mean, t_std = t_ckpt["mean"].to(dev), t_ckpt["std"].to(dev)
            to_teacher = lambda x, m=t_mean, sd=t_std: (x * std + mean - m) / sd
            ta = t_ckpt["args"]
            if has_val and (ta["val_frac"], ta["test_frac"], ta["seed"]) != (args.val_frac, args.test_frac, args.seed):
                print(f"WARNING: teacher {path} used a different split, so it may have trained on this run's val / "
                      "test images and the local scores will be optimistic")
            _, acc = evaluate(teacher, to_teacher(x_val), y_val, 1000, amp)
            print(f"teacher: {path}, ResNet-{2 * sum(ta['blocks']) + 2} width {ta['width']}, "
                  f"{sum(q.numel() for q in teacher.parameters()):,} params, train seed {ta.get('train_seed', ta['seed'])}, "
                  f"val acc {acc:.4f}")
            teachers.append((teacher, to_teacher, acc))

    def teacher_logits(xb):
        """Average of the teachers' logits on the student's (normalized) batch."""
        return torch.stack([t(tt(xb)).float() for t, tt, _ in teachers]).mean(0)

    t_val_acc, t_val_accs = float("nan"), [acc for _, _, acc in teachers]
    if teachers and has_val:
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            ens = torch.cat([teacher_logits(x_val[i:i + 1000]) for i in range(0, len(x_val), 1000)])
        t_val_acc = (ens.argmax(1) == y_val).float().mean().item()
    if teachers:
        print(f"distilling from {len(teachers)} teacher(s), ensemble val acc {t_val_acc:.4f}; "
              f"kd alpha {args.kd_alpha}, temp {args.kd_temp}")

    params = list(model.parameters())
    decay = [q for q in model.parameters() if q.ndim > 1]
    no_decay = [q for q in model.parameters() if q.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=0.9, nesterov=True)
    steps_per_epoch = math.ceil(len(x_train) / args.batch)
    total, warm = args.epochs * steps_per_epoch, round(args.warmup * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    args.out.mkdir(parents=True, exist_ok=True)
    history, best_acc, best_epoch, stopped_early, train_start = [], -1.0, 0, False, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        perm = torch.randperm(len(x_train), device=dev)
        loss_sum = torch.zeros((), device=dev)
        correct = torch.zeros((), device=dev)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            # jitter + grayscale + sharpness + RandAugment on [0, 1] pixels -> normalize -> flip + crop -> Cutout -> Mixup / CutMix
            xb = color_jitter(x_train[idx], *args.jitter)
            if args.grayscale:
                xb = random_grayscale(xb, args.grayscale)
            if args.sharpness:
                xb = sharpen(xb, args.sharpness, args.sharpness_prob)
            if args.randaugment[0]:
                xb = rand_augment(xb, *args.randaugment)
            xb = augment(norm(xb))
            if args.zoom != [1.0, 1.0]:
                xb = zoom(xb, *args.zoom)
            if args.cutout:
                xb = cutout(xb, args.cutout)
            ya = yb = y_train[idx]
            lam = 1.0
            if args.mix != "none" and torch.rand(()).item() < args.mix_prob:
                xb, ya, yb, lam = mix_batch(xb, ya, args.mix, args.mix_alpha)
            xb = xb.contiguous(memory_format=torch.channels_last)
            if teachers:  # the targets do not depend on the student's weights: computed once per batch
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    t_logits = teacher_logits(xb)

            def forward_loss():
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(xb).float()
                loss = criterion(logits, ya) if lam == 1.0 else lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                if teachers:
                    loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd_loss(logits, t_logits, args.kd_temp)
                return logits, loss

            logits, loss = forward_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.sam:  # SAM: gradient at the worst nearby weights, applied to the original weights
                offsets = sam_perturb(params, args.sam)
                opt.zero_grad(set_to_none=True)
                prev_momentum = set_bn_momentum(model, 0.0)  # BN running stats come from the first pass only
                forward_loss()[1].backward()
                for m, mom in prev_momentum:
                    m.momentum = mom
                sam_restore(params, offsets)
            opt.step()
            sched.step()
            loss_sum += loss.detach() * len(idx)
            pred = logits.argmax(1)  # on a mixed batch, credit each image's two labels by their weights
            correct += lam * (pred == ya).sum() + (1 - lam) * (pred == yb).sum()
        val_loss, val_acc = evaluate(model, x_val, y_val, 1000, amp)
        h = {"epoch": epoch, "train_loss": loss_sum.item() / len(x_train), "train_acc": correct.item() / len(x_train),
             "val_loss": val_loss, "val_acc": val_acc, "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t_ep}
        history.append(h)
        if not has_val or val_acc > best_acc:  # without val, every epoch overwrites best.pt
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": saved_args, "mean": mean.cpu(), "std": std.cpu()}, args.out / "best.pt")
        print(f"epoch {epoch:3d}  train {h['train_loss']:.4f} / {h['train_acc']:.4f}  "
              f"val {val_loss:.4f} / {val_acc:.4f}  lr {h['lr']:.4f}  {h['seconds']:.1f}s", flush=True)
        if has_val and args.patience and epoch - best_epoch >= args.patience:
            stopped_early = True
            print(f"early stop: no val improvement in {args.patience} epochs (best epoch {best_epoch})")
            break
    train_seconds = time.time() - train_start

    best = torch.load(args.out / "best.pt", map_location=dev)
    model.load_state_dict(best["model"])
    test_loss, test_acc = evaluate(model, x_test, y_test, 1000, amp)
    _, test_acc_tta = evaluate(model, x_test, y_test, 1000, amp, tta=True)
    preds = predict(model, x_kag, 1000, amp).cpu().numpy()
    preds_tta = predict(model, x_kag, 1000, amp, tta=True).cpu().numpy()
    write_submission(args.out / "submission.csv", kaggle_ids, preds)
    write_submission(args.out / "submission_tta.csv", kaggle_ids, preds_tta)

    nan_to_none = lambda v: None if math.isnan(v) else v
    summary = {"depth": depth, "params": n_params, "best_epoch": best["epoch"],
               "best_val_acc": best_acc if has_val else None, "epochs_run": len(history), "stopped_early": stopped_early,
               "test_loss": nan_to_none(test_loss), "test_acc": nan_to_none(test_acc),
               "test_acc_tta": nan_to_none(test_acc_tta), "train_seconds": train_seconds,
               "split": {"train": len(tr), "val": len(va), "test": len(te)},
               "teacher_val_acc": nan_to_none(t_val_acc) if teachers else None,
               "teacher_val_accs": [nan_to_none(a) for a in t_val_accs] if teachers else None,
               "kaggle_pred_counts": np.bincount(preds, minlength=10).tolist(),
               "kaggle_tta_changed": int((preds != preds_tta).sum()),
               "args": saved_args}
    (args.out / "history.json").write_text(json.dumps(history, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(history, args.out / "curves.png")
    print(f"best epoch {best['epoch']}: val {best_acc if has_val else float('nan'):.4f}  local test {test_acc:.4f} / {test_acc_tta:.4f} with TTA  "
          f"({train_seconds:.0f}s training) -> {args.out}")


if __name__ == "__main__":
    main()
