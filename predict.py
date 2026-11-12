"""Re-score a finished run's best.pt with flip test-time augmentation, without retraining.

Rebuilds the model and the train / val / test split from the args saved in the checkpoint, prints the
local test accuracy plain and with flip TTA, and writes OUT/submission_tta.csv for Kaggle.

  python predict.py --run runs/r24_e300
  python predict.py --run runs/r24_e300 --data data   # if the run was trained with a different data path
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train


def main():
    """Rebuild a run's model and split from its checkpoint, print the local test accuracy plain and with
    flip TTA, and write submission_tta.csv next to best.pt."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run directory with best.pt")
    p.add_argument("--data", type=Path, help="data directory (default: the one saved in the checkpoint)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    dev = torch.device(a.device)
    amp = dev.type == "cuda"
    model, ckpt = train.load_run(a.run / "best.pt", dev)
    args = ckpt["args"]
    data = a.data or Path(args["data"])
    x_all, y_all, x_kaggle, kaggle_ids = train.load_images(data, data / "cache.npz")
    _, _, te = train.stratified_split(y_all, args["val_frac"], args["test_frac"], args["seed"])

    mean, std = ckpt["mean"].to(dev), ckpt["std"].to(dev)
    prep = lambda x: (torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float().div_(255) - mean) / std
    x_test, y_test, x_kag = prep(x_all[te]), torch.from_numpy(y_all[te]).to(dev), prep(x_kaggle)

    _, acc = train.evaluate(model, x_test, y_test, 1000, amp)
    _, acc_tta = train.evaluate(model, x_test, y_test, 1000, amp, tta=True)
    preds = train.predict(model, x_kag, 1000, amp).cpu().numpy()
    preds_tta = train.predict(model, x_kag, 1000, amp, tta=True).cpu().numpy()
    train.write_submission(a.run / "submission_tta.csv", kaggle_ids, preds_tta)
    print(f"{a.run}: epoch {ckpt['epoch']}, local test {acc:.4f} plain / {acc_tta:.4f} flip TTA "
          f"({len(te)} images); TTA changes {(preds != preds_tta).sum()} of {len(preds)} Kaggle predictions "
          f"-> {a.run / 'submission_tta.csv'}")
    counts = np.bincount(preds_tta, minlength=10).tolist()
    print(f"kaggle TTA pred counts: {json.dumps(counts)}")


if __name__ == "__main__":
    main()
