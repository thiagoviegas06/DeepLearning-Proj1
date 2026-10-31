# HPC runbook (A100) — for whoever picks this up

Repo is clone-ready; data is NOT tracked. Get it first:

```bash
# on HPC, from the repo clone root:
ln -s /scratch/<netid>/DL_P1/data data        # or copy the folder as data/
```

## 1) SSL pretraining (SimCLR, InfoNCE)

```bash
python ssl/simclr.py --epochs 100 --batch-size 512   # ~10 min on A100
```
Outputs `simclr_encoder.pt` (+ `simclr_state.pt` for resume). Scale to 200-300
epochs if sweeping — the schedule is saved in the state file.

## 2) Fine-tune from the SSL encoder

```bash
python ssl/train.py --epochs 200 --init simclr_encoder.pt --tag simclr-ft
```
Data: expects `./data` with train_images/, train_labels.csv, test_images/, test.csv.
Supervised aug already includes the photometric fix (ColorJitter 0.3 + sharpness).

## 3) The combined recipe (our 0.90+ shot)

Their root `train.py` recipe (ResNet-24: `--blocks 4 4 3 --width 64 --kernel 3`,
200 ep, Cutout 16, CutMix, LS 0.1 → 0.886 Kaggle) **+ stronger photometric arm**:
bump `--jitter 0.4 0.4 0.4` and add sharpness augmentation. Measured test-vs-train
shift (brighter/contrastier/sharper) is in ssl/README.md — every point of photometric
robustness added so far shrank the val→Kaggle gap (10.7 → 8.1 across our runs).

## 4) Submit

```bash
python ssl/predict.py --ckpt best_simclr-ft.pt --out submission_ssl.csv
kaggle competitions submit -c cifar-10-res-net-architecture-challenge-dl \
  -f submission_ssl.csv -m "SSL init + combined recipe"
```

Quota: 5/day, resets ~20:00 EDT (UTC midnight).

## Status @ this commit

- Supervised + photometric aug: 94.64% val → 0.855 Kaggle
- SSL pretrain 100ep + fine-tune (this laptop, MPS): **95.32% val** (final) →
  Kaggle scored 0.844 (gap didn't transfer — see README analysis)
- Team best: ResNet-24 CutMix 0.886 (96.66% val)
- Leaderboard top: 0.909
