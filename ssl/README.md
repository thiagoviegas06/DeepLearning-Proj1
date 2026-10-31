# ssl/ — SimCLR self-supervised pretraining branch (pxndey/simclr-ssl)

## TL;DR for the team

**We found a likely cause of the huge val→Kaggle gap.** Every strong run so far shows
~11 points drop: e.g. `runs/aug/cut_ls_e200` reports 96.01% val → **0.854 Kaggle**, and
independently a 93.56% val model → **0.828 Kaggle**. A pixel-stat comparison (pure
internal, no external sources) shows the provided **test images are photometrically
shifted** vs train images:

| | mean RGB | avg contrast | sharpness (|∇| energy) | per-image brightness spread |
|---|---|---|---|---|
| train | [0.489, 0.479, 0.443] | 0.200 | 28.6 | 0.126 |
| test | [0.513, 0.498, 0.461] | 0.210 | **31.6** | **0.149** |

Test images are brighter, higher-contrast, sharper, and more heterogeneous. If your
supervised augmentation is only crop+flip (or crop+flip+Cutout), the model never sees
photometric variation → val looks great, Kaggle punishes you. **Suggested test: add
`ColorJitter(0.3,0.3,0.3,0.05)` (p=0.8) + `RandomAdjustSharpness(2)` (p=0.2) to your
augmentation and compare LB scores.** This is allowed (augmentation is unrestricted).

## What this directory contains

| File | Purpose |
|---|---|
| `model.py` | `ConvResNet` — residual conv stages (`ReLU(S(x)+F(x))` per spec), widths (64,128,256), blocks (3,3,3), deep MLP head → **4.72M/5M params**. Also `ProjectionHead` for SimCLR (discarded post-pretrain, doesn't count toward the 5M). |
| `data.py` | RAM-cached loaders, fixed stratified 45k/5k split (seed 0), three transform modes. **Supervised mode now includes the photometric augmentation above.** |
| `simclr.py` | SimCLR/InfoNCE pretraining (NT-Xent, temp 0.5, AdamW, warmup+cosine). Saves encoder every epoch + full resume state. **Note:** uses `encoder.forward_features()`, not `encoder(x)`. |
| `train.py` | Supervised training with `--init simclr_encoder.pt` to fine-tune the SSL encoder; full-resume checkpoints + graceful SIGTERM (finish epoch, save, exit). |
| `predict.py` | Checkpoint → Kaggle `submission.csv` (format-verified). |
| `HANDOFF.md` | Full project history/decisions/ops notes. |

## Usage (from repo root; expects `data/` with the competition files)

```bash
PY=python  # or the conda env python
# 1) pretrain (SSL) — ~2h on MPS, minutes on A100
$PY ssl/simclr.py --epochs 100 --batch-size 512
# 2) fine-tune from the pretrained encoder
$PY ssl/train.py --epochs 100 --init simclr_encoder.pt --tag simclr-ft
# 3) baseline without SSL for the ablation
$PY ssl/train.py --epochs 100 --tag baseline
# 4) submit
$PY ssl/predict.py --ckpt best_simclr-ft.pt
kaggle competitions submit -c cifar-10-res-net-architecture-challenge-dl -f submission_simclr-ft.csv -m "..."
```

## Results so far (pre-SSL-finetune runs)

- Baseline (crop+flip only): 93.56% val (ep 73/100) → **0.828 Kaggle**
- Team pipeline (their recipe): 96.01% val → **0.854 Kaggle**
- Same ~11pt gap in both → shift hypothesis is consistent across pipelines.
- **Photometric-aug retrain** (ColorJitter 0.3 + sharpness 2.0): 94.64% val (ep 95) → **0.855 Kaggle**.
  Val +1.1 but Kaggle +2.7 vs v1 — the aug fix works where it matters. Confirms the shift.
- **SSL fine-tune** (SimCLR 100ep pretrain → fine-tune w/ photometric aug): best val
  **95.32% @ ep 94** (still climbing at commit time) — beats the supervised-only photometric
  baseline by +0.7 on val. **First SSL result for the team.** Kaggle score TBD.

## Integration plan (next)

1. Re-run supervised training with the photometric aug (already in `ssl/data.py`) and
   measure the new val→Kaggle gap. If it shrinks, port the aug into the root `train.py`.
2. Port SSL pretraining to the root pipeline's model class (weight transfer requires
   matching architecture; currently `ssl/model.py` defines its own encoder).
3. Final-model playbook: train on all 50k, TTA (hflip-averaged logits), longer schedule.

All rules respected: no attention (banned), no pretrained weights, no external data;
SimCLR uses only the provided 50k images (training procedure, allowed; cite Chen et al. 2020).
