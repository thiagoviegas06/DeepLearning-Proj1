"""Baseline ResNet with 5x5 convolutions for the Kaggle CIFAR-10 ResNet architecture challenge.

Data (data/): 50,000 labeled 32x32 RGB training images (5,000 per class) and 2,000 unlabeled
Kaggle test images. The labeled set is split per class into train / val / test (default
40,000 / 5,000 / 5,000). Val picks the best epoch; the local test split is scored once at the end.
The PNGs are decoded once into data/cache.npz, and the whole dataset lives on the device, with
random crop + flip done on the GPU.

  .venv/bin/python kaggle_resnet_cifar10/train.py                       # 100 epochs, early stopping
  .venv/bin/python kaggle_resnet_cifar10/train.py --smoke               # CPU check, tiny subset
  .venv/bin/python kaggle_resnet_cifar10/train.py --blocks 2 2 2 --width 16

Writes OUT/history.json, summary.json, curves.png, best.pt and submission.csv (id,label).
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


@torch.no_grad()
def evaluate(model, x, y, batch, amp):
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(x), batch):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x[i:i + batch]).float()
        loss += F.cross_entropy(logits, y[i:i + batch], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    return loss / len(x), correct / len(x)


@torch.no_grad()
def predict(model, x, batch, amp):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            out.append(model(x[i:i + batch]).argmax(1))
    return torch.cat(out)


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
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--max-params", type=int, default=5_000_000, help="refuse larger models (0 = no limit)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20,
                   help="stop after this many epochs without a new best val accuracy (0 = off)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup", type=float, default=1.0, help="linear warmup, in epochs")
    p.add_argument("--jitter", type=float, nargs=3, default=[0.2, 0.2, 0.2], metavar=("B", "C", "S"),
                   help="color jitter strength for brightness, contrast, saturation (0 0 0 = off)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="seeds both the split and training")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="1 epoch on 1,000 train images, tiny eval sets")
    args = p.parse_args()
    if args.smoke:
        args.epochs = 1

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
    if args.max_params and n_params > args.max_params:
        raise SystemExit(f"{n_params:,} params exceeds --max-params {args.max_params:,}")

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
            xb = augment(norm(color_jitter(x_train[idx], *args.jitter))).contiguous(memory_format=torch.channels_last)
            yb = y_train[idx]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach() * len(idx)
            correct += (logits.argmax(1) == yb).sum()
        val_loss, val_acc = evaluate(model, x_val, y_val, 1000, amp)
        h = {"epoch": epoch, "train_loss": loss_sum.item() / len(x_train), "train_acc": correct.item() / len(x_train),
             "val_loss": val_loss, "val_acc": val_acc, "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t_ep}
        history.append(h)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args) | {"data": str(args.data),
                        "out": str(args.out)}, "mean": mean.cpu(), "std": std.cpu()}, args.out / "best.pt")
        print(f"epoch {epoch:3d}  train {h['train_loss']:.4f} / {h['train_acc']:.4f}  "
              f"val {val_loss:.4f} / {val_acc:.4f}  lr {h['lr']:.4f}  {h['seconds']:.1f}s", flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            stopped_early = True
            print(f"early stop: no val improvement in {args.patience} epochs (best epoch {best_epoch})")
            break
    train_seconds = time.time() - train_start

    best = torch.load(args.out / "best.pt", map_location=dev)
    model.load_state_dict(best["model"])
    test_loss, test_acc = evaluate(model, x_test, y_test, 1000, amp)
    preds = predict(model, x_kag, 1000, amp).cpu().numpy()
    with open(args.out / "submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(zip(kaggle_ids.tolist(), preds.tolist()))

    summary = {"depth": depth, "params": n_params, "best_epoch": best["epoch"], "best_val_acc": best_acc,
               "epochs_run": len(history), "stopped_early": stopped_early,
               "test_loss": test_loss, "test_acc": test_acc, "train_seconds": train_seconds,
               "split": {"train": len(tr), "val": len(va), "test": len(te)},
               "kaggle_pred_counts": np.bincount(preds, minlength=10).tolist(),
               "args": vars(args) | {"data": str(args.data), "out": str(args.out)}}
    (args.out / "history.json").write_text(json.dumps(history, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(history, args.out / "curves.png")
    print(f"best epoch {best['epoch']}: val {best_acc:.4f}  local test {test_acc:.4f}  "
          f"({train_seconds:.0f}s training) -> {args.out}")


if __name__ == "__main__":
    main()
