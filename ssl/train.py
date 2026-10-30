"""Supervised training: from scratch, or fine-tuned from a SimCLR encoder.

Usage:
  python train.py --epochs 100 --tag baseline
  python train.py --epochs 100 --init simclr_encoder.pt --lr 3e-4 --tag simclr-ft
"""
import argparse
import csv
import os
import signal
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_loaders
from model import ConvResNet, count_params


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)

    stop = {"flag": False}
    def on_term(signum, frame):
        stop["flag"] = True
        print(f"\n[signal {signum}] stopping after current epoch (checkpoint is "
              f"saved every epoch; resume with --resume)", flush=True)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    model = ConvResNet().to(device)

    if args.init:
        ckpt = torch.load(args.init, map_location=device)
        model.load_state_dict(ckpt["encoder"])
        print(f"initialized encoder from {args.init} (epoch {ckpt.get('epoch', '?')})")
    print(f"device: {device} | trainable parameters: {count_params(model):,} "
          f"(limit 5,000,000)")

    train_loader, val_loader = get_loaders(batch_size=args.batch_size,
                                           num_workers=args.workers, seed=args.seed)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    epochs = args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=epochs * len(train_loader),
        pct_start=0.05, anneal_strategy="cos")

    start_epoch, best = 1, 0.0
    state_path = f"state_{args.tag}.pt"
    if args.resume and os.path.exists(state_path):
        s = torch.load(state_path, map_location=device)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        start_epoch, best = s["epoch"] + 1, s["best"]
        print(f"resumed from {state_path} (epoch {s['epoch']}, best {best*100:.2f}%)")

    for ep in range(start_epoch, epochs + 1):
        model.train()
        t0, loss_sum, nb = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item(); nb += 1
        val_acc = evaluate(model, val_loader, device)
        marker = ""
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "val_acc": best,
                        "epoch": ep, "args": vars(args)}, f"best_{args.tag}.pt")
            marker = "  <- best"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best},
                   state_path)
        if stop["flag"]:
            print(f"epoch {ep:3d}/{epochs} | loss {loss_sum/nb:.4f} | "
                  f"val acc {val_acc*100:.2f}%{marker} | {time.time()-t0:.0f}s "
                  f"[SIGTERM: saved and stopping]", flush=True)
            print(f"\ngraceful stop at epoch {ep} | best val acc {best*100:.2f}% "
                  f"({count_params(model):,} parameters)", flush=True)
            sys.exit(0)
        print(f"epoch {ep:3d}/{epochs} | loss {loss_sum/nb:.4f} | "
              f"val acc {val_acc*100:.2f}%{marker} | {time.time()-t0:.0f}s", flush=True)

    print(f"\nbest validation accuracy: {best*100:.2f}% "
          f"({count_params(model):,} parameters)")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--init", type=str, default=None, help="SimCLR encoder checkpoint")
    p.add_argument("--tag", type=str, default="baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true",
                   help="resume from state_{tag}.pt if present")
    train(p.parse_args())
