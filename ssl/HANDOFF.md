# Project 1 Handoff — CIFAR-10 ResNet Kaggle Competition

**Updated:** Oct 30, 2026, ~2:30 AM — all training STOPPED, GPU idle, machine clean.
**Deadline:** competition ends Nov 6, 11:59 PM · report due Nov 9 (Brightspace, ≤8 pages)
**Worth:** 20% of grade. Teams ≤3. Top-20% = +1 pt, top team +2 pts.

## The task

Modified ResNet (any **conv** net with skip connections), ≤ **5,000,000 trainable
params**, trained **from scratch** on competition-provided data only.
Highest accuracy on 2,000 held-out test images.

**Rules:**
- ❌ NO attention (prof banned it Oct 29 — killed the conv+ViT hybrid idea)
- ❌ No pretrained weights, no external datasets, eval labels must not be inferred externally
- ✅ Self-supervised pretraining (SimCLR/InfoNCE) on the provided 50k is allowed (training procedure, not architecture; projection head discarded → doesn't count toward 5M)
- ✅ Any augmentation/optimizer/regularizer/schedule/distillation; external code with citation

## Results so far (leaderboard facts)

| Artifact | Val acc | Public LB |
|---|---|---|
| `best_baseline_ep73.pt` (recipe v1: crop+flip only) | **93.56%** | **82.8%** |
| Team's earlier submissions (unknown models) | ? | 79.3%, 81.1% |
| Leaderboard leader ("Henry") | ? | **89.2%** |

**Submission quota: 3 remaining today (resets daily ~UTC midnight).** Kaggle accepted
our CSV format fine (submission ref 57788824).

## THE KEY FINDING (validated Oct 30, ~2 AM): val→test gap of 10.7 pts

93.56% val vs 82.8% public is NOT normal generalization. Measured distribution shift
between the provided train images and test images (pure internal comparison, rule-safe):

| | mean RGB | avg contrast | sharpness (|∇| energy) | per-image brightness spread |
|---|---|---|---|---|
| train (2k sample) | [0.489, 0.479, 0.443] | 0.200 | 28.6 | 0.126 |
| test (all 2k) | [0.513, 0.498, 0.461] | 0.210 | **31.6** | **0.149** |

Test images are **brighter, higher-contrast, sharper, more heterogeneous**. Our v1
recipe had **zero photometric augmentation** (only crop+flip) → model tuned to train's
photometric regime. Supporting evidence: team's weaker models only ~1.7 pts below ours
on LB despite 12 pts lower val — everyone is compressed by this shift; Henry's 89.2%
shows ~89 is reachable on test.

**Fix applied to `data.py` `sup_train_tf` (NOT yet trained):**
- added `ColorJitter(0.3, 0.3, 0.3, 0.05)` @ p=0.8
- added `RandomAdjustSharpness(2.0)` @ p=0.2

## Repo state (~/nyu/f26/dl/project1)

```
competition_data/       official data: 50k train + labels, 2k test, CSVs (verified)
data.zip                backup archive (deletable, 120MB)
model.py                ConvResNet (NO attention): stem + 3 res-stages + GAP + deep
                        MLP head. Default widths(64,128,256) blocks(3,3,3) = 4.72M/5M.
                        Block = ReLU(S(x)+F(x)) per spec. ProjectionHead for SimCLR.
data.py                 RAM-cached loaders; stratified 45k/5k split seed 0; three
                        transform modes (sup w/ photometric aug NOW, contrastive, eval).
                        Stats: mean (0.4907,0.4818,0.4465) std (0.2450,0.2407,0.2600).
simclr.py               NT-Xent pretrain. USES encoder.forward_features() NOT encoder(x).
                        Saves simclr_encoder.pt + simclr_state.pt (full resume) each ep.
train.py                Supervised: SGD nesterov lr 0.05, OneCycle, label smoothing 0.1,
                        wd 5e-4, batch 256. Saves best_{tag}.pt + state_{tag}.pt (full
                        resume: model+opt+sched+epoch) EVERY epoch. Graceful SIGTERM
                        handling (finishes epoch, saves, exits 0) + --resume flag.
predict.py              checkpoint -> submission CSV, format + class-dist checks. WORKS.
logs/                   run logs (baseline.log, simclr.log, photometric.log, stop.log)
HANDOFF.md              this file
```

**Checkpoints on disk:**
- `best_baseline_ep73.pt` — 93.56% val / 82.8% public (v1 recipe, died at ep 73 of 100)
- `simclr_encoder_ep78.pt` — pretrain encoder at ep 78/100 (v1, interrupted)
- `best_baseline.pt` + `state_baseline.pt` — v2 rerun reached ep ~45, best 89.74% (deterministic replay of v1)
- `simclr_encoder.pt` + `simclr_state.pt` — v2 pretrain, ep ~45-ish, full resume state
- `submission_baseline_ep73.csv` — the 82.8% submission

## Operational lessons (painfully earned Oct 29-30)

1. **Kaggle API can't bulk-download this comp** (bundle endpoint 404s; per-file only;
   parallel fetch → ~11h 429 ban). The notebook-zip trick worked (see git history of
   this file). Everything's downloaded now; don't touch the API for data again.
2. **OpenCode server restarts kill tool-spawned background shells.** Long jobs must be
   launched `nohup ... > logs/x.log 2>&1 &` from the project dir — they survive.
3. **Never stack/kill heavy GPU jobs carelessly**: killing two MPS processes under
   memory pressure then immediately launching predict.py froze the whole laptop
   (23/24 GB used, swap thrash). Sequence: kill → verify memory free → launch.
4. Two concurrent trainings ≈ 2.5 min/epoch each; one solo ≈ 1.5 min/epoch.
   `caffeinate` (killed now) prevents sleep on AC — re-arm for long runs: `caffeinate -s &`.
5. Runs are seed-0 deterministic: same command → identical val trajectory (verified).
6. Don't run heavy jobs when battery is low/flaky — user had a 2% + charger-flakiness scare.

## Next steps (priority order)

1. **Retrain with the photometric-aug recipe** (the fix above) — `python train.py
   --epochs 100 --tag photometric` (solo GPU ~1.5-2h). Expect val ~92-94% AND much
   smaller val→test gap. Submit when done (quota: 3 left today).
2. **If gap shrinks**: also fine-tune the SimCLR encoder (resume ep 45 state or ep 78
   backup) with the new aug → compare. SimCLR + photometric aug may stack.
3. **Final-model tricks for the last submissions**: retrain on ALL 50k (fold val in)
   with the winning recipe; TTA (average logits over hflip) at predict time; consider
   150-200 epochs if time allows.
4. **Diagnostics if gap persists**: model confidence histogram test-vs-val (OOD check);
   try normalizing test with its OWN stats as an experiment (cheap, rule-safe);
   per-class error breakdown on val.
5. **Report + GitHub repo** (due Nov 9): repo link on page 1, methodology (architecture
   evolution incl. the attention ban, the photometric-shift discovery is GOLD for the
   methodology section), final test acc + param count, citations (Chen et al. 2020 for
   SimCLR; kuangliu/pytorch-cifar if any code was borrowed — it wasn't so far).
6. Keep HANDOFF.md updated at each milestone.

## Env cheatsheet

```bash
PY=/Users/pxndey/miniconda3/envs/summer-sprint/bin/python
$PY train.py --epochs 100 --tag NAME [--resume]           # supervised
$PY train.py --epochs 100 --init simclr_encoder.pt --tag FT [--resume]
$PY simclr.py --epochs 100 --batch-size 512 [--resume]
$PY predict.py --ckpt best_X.pt                           # -> submission CSV
kaggle competitions submit -c cifar-10-res-net-architecture-challenge-dl \
  -f submission_X.csv -m "message"
kaggle competitions submissions -c cifar-10-res-net-architecture-challenge-dl
```
