"""Self-supervised SimCLR-style pretraining (InfoNCE / NT-Xent loss).

Pretrains the HybridResNet encoder from scratch on the 50k provided training
images only (no external data, no pretrained weights — rule-compliant).
The projection head is discarded afterwards; only the encoder is fine-tuned.

Reference: Chen et al., "A Simple Framework for Contrastive Learning of
Visual Representations", ICML 2020.
"""
import argparse
import math
import os
import signal
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_contrastive_loader
from model import ConvResNet, ProjectionHead, count_params


def nt_xent(z1, z2, temperature=0.5):
    """NT-Xent loss. z1, z2: (B, D) normalized projections of the two views."""
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2B, D)
    sim = z @ z.t() / temperature                        # (2B, 2B)
    n = z1.size(0)
    # positives: i <-> i+n ; mask out self-similarity
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def pretrain(epochs=100, batch_size=512, lr=3e-4, temperature=0.5,
             weight_decay=1e-6, warmup=10, out="simclr_encoder.pt",
             resume=False, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)

    stop = {"flag": False}
    def on_term(signum, frame):
        stop["flag"] = True
        print(f"\n[signal {signum}] stopping after current epoch (checkpoint is "
              f"saved every epoch; resume with --resume)", flush=True)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    encoder = ConvResNet().to(device)
    proj = ProjectionHead().to(device)
    print(f"device: {device} | encoder params: {count_params(encoder):,}")

    loader = get_contrastive_loader(batch_size=batch_size)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(proj.parameters()),
                            lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / warmup_steps if s < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / (total_steps - warmup_steps)))
    ))

    start_ep = 1
    state_path = "simclr_state.pt"
    if resume and os.path.exists(state_path):
        s = torch.load(state_path, map_location=device)
        encoder.load_state_dict(s["encoder"]); proj.load_state_dict(s["proj"])
        opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start_ep = s["epoch"] + 1
        print(f"resumed from {state_path} (epoch {s['epoch']})")

    for ep in range(start_ep, epochs + 1):
        encoder.train(); proj.train()
        t0, loss_sum, nb = time.time(), 0.0, 0
        for (v1, v2), _ in loader:
            v1, v2 = v1.to(device), v2.to(device)
            loss = nt_xent(proj(encoder.forward_features(v1)),
                           proj(encoder.forward_features(v2)), temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item(); nb += 1
        print(f"epoch {ep:3d}/{epochs} | nt-xent {loss_sum/nb:.4f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        torch.save({"encoder": encoder.state_dict(), "epoch": ep}, out)
        torch.save({"encoder": encoder.state_dict(), "proj": proj.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep}, state_path)
        if stop["flag"]:
            print(f"graceful stop at epoch {ep} | encoder saved to {out}", flush=True)
            sys.exit(130)   # non-zero so a chained `&& train.py` does NOT start
    print(f"encoder saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--out", type=str, default="simclr_encoder.pt")
    p.add_argument("--resume", action="store_true",
                   help="resume from simclr_state.pt if present")
    a = p.parse_args()
    pretrain(**vars(a))
