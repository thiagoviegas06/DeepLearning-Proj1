"""Review sheet for a trained run: its least confident predictions, as image pages and a CSV.

Loads RUN/best.pt, scores the images with flip TTA (the same way submission_tta.csv is made), sorts
them by confidence (the top softmax probability) and writes, into OUT/:

  low_confidence.csv      every image, least confident first: id, prediction, confidence, runner-up, its probability
  page_01.png, page_02...  the N least confident images, PER_PAGE per page, titled "id: pred conf | runner-up"
  confidence_hist.png      histogram of the confidence over all images
  summary.json             counts below a few confidence thresholds, predicted class counts

With --split kaggle (default) the images are the 2,000 competition images, which have no labels.
With --split val or --split test the images are the run's own local split, which has labels: the
titles then show the true class and wrong predictions are framed in red, so the same review can be
done where the answer is known. (The split is rebuilt from the run's --seed / --val-frac / --test-frac.)

  python inspect_predictions.py --run runs/r24_kd_t3_e300_full                    # Kaggle, 192 least confident
  python inspect_predictions.py --run runs/r24_kd_t3_e300 --split val --n 96     # local val, with the truth
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train  # noqa: E402

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def pick_device(name):
    """torch.device for `name`, or the best available one (CUDA > MPS > CPU) when name is empty."""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def main():
    """Score the chosen split with flip TTA, rank the images least confident first and write the review
    sheet (low_confidence.csv, confidence_hist.png, page_NN.png, summary.json) to --out."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run directory holding best.pt and summary.json")
    p.add_argument("--data", type=Path, default=HERE / "data")
    p.add_argument("--split", choices=["kaggle", "val", "test"], default="kaggle")
    p.add_argument("--n", type=int, default=192, help="how many least-confident images to draw")
    p.add_argument("--per-page", type=int, default=96)
    p.add_argument("--out", type=Path, default=None, help="default: analysis/<run name>_<split>/")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    out = args.out or HERE / "analysis" / f"{args.run.name}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(args.device)

    cache = np.load(args.data / "cache.npz")
    if args.split == "kaggle":
        images, ids, truth = cache["x_kaggle"], cache["kaggle_ids"], None
    else:
        run_args = json.load(open(args.run / "summary.json"))["args"]
        if not run_args.get("val_frac") or not run_args.get("test_frac"):
            raise SystemExit(f"{args.run} was trained on all 50k images: it has no local {args.split} split")
        tr, va, te = train.stratified_split(cache["y_train"], run_args["val_frac"], run_args["test_frac"], run_args["seed"])
        idx = va if args.split == "val" else te
        images, ids, truth = cache["x_train"][idx], idx, cache["y_train"][idx]

    model, ckpt = train.load_run(args.run / "best.pt", dev)
    mean, std = ckpt["mean"].to(dev).reshape(1, 3, 1, 1), ckpt["std"].to(dev).reshape(1, 3, 1, 1)
    x = (torch.tensor(images).permute(0, 3, 1, 2).float().div(255).to(dev) - mean) / std
    amp = dev.type == "cuda"
    with torch.no_grad():
        probs = torch.softmax(train.logits_of(model, x.contiguous(memory_format=torch.channels_last), 500, amp, tta=True), 1).cpu().numpy()
    order = np.argsort(probs, 1)[:, ::-1]
    pred, second = order[:, 0], order[:, 1]
    conf, p2 = probs[np.arange(len(probs)), pred], probs[np.arange(len(probs)), second]
    rank = np.argsort(conf)

    with open(out / "low_confidence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred", "pred_name", "confidence", "second", "second_name", "p_second"] + (["true", "true_name", "correct"] if truth is not None else []))
        for i in rank:
            row = [int(ids[i]), int(pred[i]), CLASSES[pred[i]], f"{conf[i]:.4f}", int(second[i]), CLASSES[second[i]], f"{p2[i]:.4f}"]
            if truth is not None:
                row += [int(truth[i]), CLASSES[truth[i]], int(pred[i] == truth[i])]
            w.writerow(row)

    summary = {"run": str(args.run), "split": args.split, "n_images": int(len(probs)),
               "below_0.5": int((conf < 0.5).sum()), "below_0.7": int((conf < 0.7).sum()), "below_0.9": int((conf < 0.9).sum()),
               "mean_confidence": float(conf.mean()), "pred_counts": dict(zip(CLASSES, np.bincount(pred, minlength=10).tolist()))}
    if truth is not None:
        summary["accuracy"] = float((pred == truth).mean())
        summary["errors"] = int((pred != truth).sum())
        summary["errors_among_below_0.7"] = int(((pred != truth) & (conf < 0.7)).sum())
        summary["accuracy_above_0.9"] = float((pred == truth)[conf >= 0.9].mean())
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(conf, bins=40, range=(0, 1))
    ax.set(xlabel="confidence (top softmax probability, flip TTA)", ylabel="images", title=f"{args.run.name}, {args.split}")
    fig.tight_layout(); fig.savefig(out / "confidence_hist.png", dpi=120); plt.close(fig)

    show = rank[:args.n]
    ncol = 12
    for page, start in enumerate(range(0, len(show), args.per_page), 1):
        sel = show[start:start + args.per_page]
        nrow = int(np.ceil(len(sel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.55, nrow * 1.85))
        for ax, i in zip(np.atleast_1d(axes).flat, sel):
            ax.imshow(images[i])
            title = f"{ids[i]}: {CLASSES[pred[i]][:5]} {conf[i]:.2f} | {CLASSES[second[i]][:5]} {p2[i]:.2f}"
            color = "black"
            if truth is not None:
                title += f"\ntrue {CLASSES[truth[i]]}"
                color = "black" if pred[i] == truth[i] else "red"
            ax.set_title(title, fontsize=6.5, color=color)
            ax.set_xticks([]); ax.set_yticks([])
            if truth is not None and pred[i] != truth[i]:
                for s in ax.spines.values():
                    s.set_edgecolor("red"); s.set_linewidth(2)
            else:
                ax.axis("off")
        for ax in list(np.atleast_1d(axes).flat)[len(sel):]:
            ax.axis("off")
        fig.suptitle(f"{args.run.name} on {args.split}: least confident predictions {start + 1}-{start + len(sel)} of {len(probs)} "
                     f"(id: prediction confidence | runner-up probability)", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"page_{page:02d}.png", dpi=110); plt.close(fig)

    print(json.dumps(summary, indent=1))
    print(f"wrote {out}/low_confidence.csv, confidence_hist.png and {page} page(s) of {args.per_page}")


if __name__ == "__main__":
    main()
