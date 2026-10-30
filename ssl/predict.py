"""Generate a Kaggle submission from a trained checkpoint.

Usage:
  python predict.py --ckpt best_baseline.pt [--out submission_baseline.csv]

Writes <out> with columns id,label (id 0..1999, label 0..9) matching
sample_submission.csv exactly.
"""
import argparse
import csv

import torch

from data import get_test_loader
from model import ConvResNet, count_params


def predict(ckpt_path, out_path, device=None, batch_size=512):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ConvResNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} (val_acc {ckpt.get('val_acc', '?'):.4f}, "
          f"epoch {ckpt.get('epoch', '?')}, {count_params(model):,} params)")

    preds = {}
    with torch.no_grad():
        for x, ids in get_test_loader(batch_size=batch_size):
            logits = model(x.to(device))
            classes = logits.argmax(1).cpu().tolist()
            for i, c in zip(ids, classes):
                preds[int(i)] = int(c)

    ids = sorted(preds)
    assert ids == list(range(2000)), f"unexpected ids: {ids[:5]}..."
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i in ids:
            w.writerow([i, preds[i]])
    # class distribution sanity check
    from collections import Counter
    print(f"wrote {out_path} | label distribution: "
          f"{dict(sorted(Counter(preds.values()).items()))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="best_baseline.pt")
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()
    out = a.out or f"submission_{a.ckpt.replace('best_', '').replace('.pt', '')}.csv"
    predict(a.ckpt, out)
