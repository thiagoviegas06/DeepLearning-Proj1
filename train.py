"""CIFAR-style ResNet for the Kaggle CIFAR-10 ResNet architecture challenge (<= 5M parameters).

Depth, width and kernel size are flags; the default is the 5x5-conv ResNet-20 baseline (3.0M
params). Data (data/): 50,000 labeled 32x32 RGB training images (5,000 per class) and 2,000
unlabeled Kaggle test images. The labeled set is split per class into train / val / test (default
40,000 / 5,000 / 5,000). Val picks the best epoch; the local test split is scored once at the end.
With --val-frac 0 --test-frac 0 the model trains on all 50,000 and the last epoch is kept.
The PNGs are decoded once into data/cache.npz, and the whole dataset lives on the device. All
augmentation (color jitter, sharpness, flip, pad-and-crop, Cutout, Mixup / CutMix) runs on the GPU
per batch. The local test split and the Kaggle images are scored both plain and with flip TTA.

  python train.py                                   # 5x5 ResNet-20 baseline, 100 epochs, early stopping
  python train.py --smoke                           # CPU check, tiny subset
  python train.py --blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 \
                  --label-smoothing 0.1 --cutout 16 --mix cutmix     # 4.7M-param recipe

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
    p.add_argument("--sharpness", type=float, default=0.0,
                   help="sharpness factor for random sharpening, e.g. 2.0 (0 = off)")
    p.add_argument("--sharpness-prob", type=float, default=0.5, help="probability that an image is sharpened")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="cross-entropy label smoothing (0 = off)")
    p.add_argument("--cutout", type=int, default=0,
                   help="side of the square hole Cutout zeroes in each training image (0 = off; 16 is standard for CIFAR)")
    p.add_argument("--mix", choices=["none", "mixup", "cutmix"], default="none", help="batch mixing augmentation")
    p.add_argument("--mix-alpha", type=float, default=1.0, help="mixing weight lambda ~ Beta(alpha, alpha)")
    p.add_argument("--mix-prob", type=float, default=1.0, help="probability that a batch is mixed")
    p.add_argument("--val-frac", type=float, default=0.1, help="0 = no val split; the last epoch is kept")
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="seeds both the split and training")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="1 epoch on 1,000 train images, tiny eval sets")
    args = p.parse_args()
    if args.smoke:
        args.epochs = 1
    has_val = args.val_frac > 0

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

    torch.manual_seed(args.seed)
    model = ResNet(tuple(args.blocks), args.width, args.kernel).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(q.numel() for q in model.parameters())
    depth = 2 * sum(args.blocks) + 2
    print(f"model: ResNet-{depth}, {args.kernel}x{args.kernel} convs, blocks {args.blocks}, "
          f"width {args.width}, {n_params:,} params, device {dev}")
    if n_params > PARAM_LIMIT:
        raise SystemExit(f"{n_params:,} parameters exceeds the competition limit of {PARAM_LIMIT:,}")
    criterion = lambda logits, y: F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)

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
            # jitter + sharpness on [0, 1] pixels -> normalize -> flip + crop -> Cutout -> Mixup / CutMix
            xb = color_jitter(x_train[idx], *args.jitter)
            if args.sharpness:
                xb = sharpen(xb, args.sharpness, args.sharpness_prob)
            xb = augment(norm(xb))
            if args.cutout:
                xb = cutout(xb, args.cutout)
            ya = yb = y_train[idx]
            lam = 1.0
            if args.mix != "none" and torch.rand(()).item() < args.mix_prob:
                xb, ya, yb, lam = mix_batch(xb, ya, args.mix, args.mix_alpha)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(xb).float()
            loss = criterion(logits, ya) if lam == 1.0 else lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
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
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args) | {"data": str(args.data),
                        "out": str(args.out)}, "mean": mean.cpu(), "std": std.cpu()}, args.out / "best.pt")
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
               "kaggle_pred_counts": np.bincount(preds, minlength=10).tolist(),
               "kaggle_tta_changed": int((preds != preds_tta).sum()),
               "args": vars(args) | {"data": str(args.data), "out": str(args.out)}}
    (args.out / "history.json").write_text(json.dumps(history, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(history, args.out / "curves.png")
    print(f"best epoch {best['epoch']}: val {best_acc if has_val else float('nan'):.4f}  local test {test_acc:.4f} / {test_acc_tta:.4f} with TTA  "
          f"({train_seconds:.0f}s training) -> {args.out}")


if __name__ == "__main__":
    main()
