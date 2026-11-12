# ResNet-24 under 5M parameters: Kaggle CIFAR-10 challenge

A CIFAR-style ResNet for a Kaggle CIFAR-10 architecture challenge whose one hard rule is **at most
5 million trainable parameters**, trained from scratch on the provided data only. The model is a
**3×3 ResNet-24 with stage widths 64/128/256 (4,697,162 parameters)** trained with SGD, a cosine
schedule, label smoothing, Cutout, CutMix and RandAugment, and distilled from an ensemble of three
18.8M-parameter teachers of the same shape (the teachers are never submitted). The whole pipeline is
one PyTorch script that keeps the dataset on the GPU and runs augmentation there too; the student
trains at about 7 s/epoch on one H100.

**Result:** 97.42% on the local held-out split (97.64% with flip TTA), level with the teacher
ensemble, and **0.907 on the Kaggle hidden test set** from the same recipe retrained on all 50,000
images. See [Results](#results).

## Competition rules this code enforces

| Rule | Where |
|---|---|
| ≤ 5,000,000 trainable parameters | `train.py` refuses to start a larger model; `model_summary.py` prints the count and the margin |
| Train from scratch on the provided CIFAR-10 only | no pretrained weights or external data anywhere; `best.pt` is written by this script |
| Clearly print the parameter count and final test accuracy | `python model_summary.py --run runs/<name>`, also called from the notebook |

## Repository layout

The training code is one script, `train.py`. Everything else either wraps it (the two Slurm scripts, the
two notebooks) or reads what it wrote (`model_summary.py`, `predict.py`, `inspect_predictions.py`).

| File | Purpose |
|---|---|
| `train.py` | model, data pipeline, GPU augmentation, training loop, distillation, Kaggle submission writer. Every hyperparameter is a flag (`python train.py -h`) |
| `model_summary.py` | model card: layer table, trainable-parameter count against the 5M limit, final test accuracy of a run (the printout the rules ask for) |
| `predict.py` | re-scores a finished run's `best.pt` with flip TTA and rewrites `submission_tta.csv`, no retraining |
| `inspect_predictions.py` | review sheets of a run's least confident predictions (image pages, CSV, histogram) |
| `train.sbatch` | **how a single run is trained on the NYU HPC**: Slurm job that runs `train.py` once with the flags in `ARGS` |
| `pipeline.sbatch` | **how the submitted model was trained**: Slurm job that runs the whole teacher-ensemble → student pipeline (eight `train.py` runs) back to back on one GPU; resumable |
| `train_notebook.ipynb` | analysis of one run: plots its curves, confusion matrix, per-class accuracy and Kaggle class counts, and prints the model card. Can also launch the run from Jupyter |
| `train_pipeline.ipynb` | analysis of the pipeline: results table, model card and plots over the finished pipeline runs. Can also run the same stages as `pipeline.sbatch` from Jupyter |
| `requirements.txt` | Python dependencies (see [Setup](#setup)) |
| `runs/<name>/` | one directory per run: `summary.json`, `history.json`, `curves.png`, `submission.csv`, `submission_tta.csv`; `best.pt` is not tracked |
| `analysis/` | `inspect_predictions.py` output for the submitted model (Kaggle images and local val split) |
| `data/` | competition data, not tracked (see [Data](#data)) |

**The notebooks are not how the results were produced.** Every run in `runs/` and every number in
[Results](#results) came from `train.sbatch` or `pipeline.sbatch` on the cluster. The notebooks exist to
look at those runs; they can launch the same `train.py` commands from an interactive GPU session, but
`train.py` and the Slurm scripts are the single source of truth for the model, data pipeline and recipe.
[How to run](#how-to-run) covers all three ways in.

## Setup

Python 3.9 or newer (3.12 was used locally and on the cluster) and the packages in `requirements.txt`:
PyTorch, NumPy, Pillow and matplotlib, plus `torchinfo` (optional, an independent cross-check of the
parameter count in `model_summary.py`) and Jupyter for the notebooks.

```bash
git clone <this repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate   # or: conda create -n deep_learning python=3.12 && conda activate deep_learning
pip install -r requirements.txt
```

`pip install torch` installs a CUDA build on Linux and an MPS (Apple GPU) build on macOS; for a specific
CUDA version use the selector at <https://pytorch.org/get-started/locally/>. Nothing else is downloaded:
there are no pretrained weights and no external datasets.

Then put the competition files in `data/` (layout in [Data](#data)) and check the plumbing end to end.
The smoke run trains for one epoch on 1,000 images, works on a CPU, and writes `runs/smoke/`:

```bash
python train.py --smoke --out runs/smoke
python model_summary.py --run runs/smoke
```

On the NYU HPC the same packages go into a Singularity overlay instead of a virtual environment; see
[On the NYU HPC](#on-the-nyu-hpc-the-slurm-scripts).

## Data

Download the competition files into `data/`:

| File | Contents |
|---|---|
| `train_images/`, `train_labels.csv` | 50,000 labeled 32×32 RGB PNGs, 5,000 per class (`id,filename,label`) |
| `test_images/`, `test.csv` | 2,000 unlabeled Kaggle test PNGs (`id,filename`) |
| `sample_submission.csv` | submission format (`id,label`) |

On first run the PNGs are decoded once into `data/cache.npz`. Later runs load that cache in seconds.

The labeled set is split **per class** with a fixed seed into:

| Split | Images | Role |
|---|---|---|
| train | 40,000 | gradient updates |
| val | 5,000 | picks the best epoch (checkpointing, early stopping) |
| test | 5,000 | scored **once** at the end, as an unbiased local estimate of the Kaggle score |

## Architecture

```
input 3×32×32
└─ stem: conv 3×3 (3→64) · BN · ReLU
   ├─ stage 1: 4 × BasicBlock(64),   32×32
   ├─ stage 2: 4 × BasicBlock(128),  16×16   (first block stride 2)
   └─ stage 3: 3 × BasicBlock(256),   8×8    (first block stride 2)
      └─ global average pool · linear(256 → 10)
```

Each `BasicBlock` is conv 3×3 · BN · ReLU · conv 3×3 · BN, plus a shortcut, then ReLU (the residual
block from the project handout). The shortcut is the identity, or a 1×1 conv + BN projection when the
stride or channel count changes. Depth is `2·sum(blocks) + 2 = 24`. The network is ResNet-18 with its
last (512-channel) stage removed and one extra block in each of the first two stages, which spends the
parameter budget on depth and width at 32×32 and 16×16 resolution.

Blocks per stage (`--blocks`), base width (`--width`) and kernel size (`--kernel`) are flags, and
`train.py` exits if the chosen configuration has more than 5M parameters. The layer table from
`python model_summary.py --blocks 4 4 3 --width 64 --kernel 3`:

| Layer | Description | Output | Params |
|---|---|---|---|
| stem | Conv 3×3 3→64 + BN + ReLU | 64×32×32 | 1,856 |
| layers.0–3 | 4 × BasicBlock 64→64, identity shortcut | 64×32×32 | 4 × 73,984 |
| layers.4 | BasicBlock 64→128, stride 2, 1×1 projection | 128×16×16 | 230,144 |
| layers.5–7 | 3 × BasicBlock 128→128, identity shortcut | 128×16×16 | 3 × 295,424 |
| layers.8 | BasicBlock 128→256, stride 2, 1×1 projection | 256×8×8 | 919,040 |
| layers.9–10 | 2 × BasicBlock 256→256, identity shortcut | 256×8×8 | 2 × 1,180,672 |
| head | global average pool + Linear 256→10 | 10 | 2,570 |
| **total** | | | **4,697,162** (302,838 under the limit) |

## Training recipe

| | |
|---|---|
| Model | `--blocks 4 4 3 --width 64 --kernel 3` (ResNet-24, 4,697,162 params) |
| Optimizer | SGD, Nesterov momentum 0.9, lr 0.1, batch 128 |
| Weight decay | 5e-4 on conv/linear weights only (none on BN or biases) |
| Schedule | 1 epoch of linear warmup, then cosine decay to 0, stepped per batch |
| Epochs | 300 for the student, 200 for each teacher, no early stopping (`--patience 0`), so the full cosine schedule runs |
| Augmentation (GPU) | color jitter (brightness/contrast/saturation ±0.2), random horizontal flip, 4-px zero-pad random crop |
| RandAugment | 1 op at magnitude 9 for the student, 2 ops for the teachers (`--randaugment 1 9` / `2 9`) |
| Random grayscale | 5% of the student's training images (`--grayscale 0.05`): the Kaggle images are grayscale 3× as often as the training set |
| SAM (optional) | off in the submitted model; `--sam 0.05` applies sharpness-aware minimization (Foret et al. 2021): the gradient is taken at the worst weights within radius 0.05 and applied to the original ones |
| EMA (optional) | off in the submitted model; `--ema 0.9995` keeps an exponential moving average of the weights (Polyak averaging) and validates, checkpoints and submits that model |
| Random zoom (optional) | off in the submitted model; `--zoom LO HI` enlarges a random window of each image by a factor drawn from U(LO, HI) after the crop (RandomResizedCrop-style; factors below 1 shrink and zero-pad). Tried as `--zoom 0.7 1.4` in round 5, see [Results](#results) |
| Distillation | three teachers, same architecture at width 128 (18,765,962 params, `--teacher-over-limit`), same data split, training seeds 0/1/2; the student matches their averaged logits with the Hinton KL loss at temperature 4, weight 0.9 (`--kd-alpha 0.9 --kd-temp 4`) |
| Test-time augmentation | predictions averaged over the image and its mirror (`submission_tta.csv`) |
| Cutout | one random 16×16 square per image zeroed after normalization (`--cutout 16`) |
| CutMix | a random box from a shuffled partner image is pasted in and the labels are mixed by area, λ ~ Beta(1, 1), every batch (`--mix cutmix`) |
| Label smoothing | 0.1 (`--label-smoothing 0.1`) |
| Normalization | per-channel mean/std computed on the training split |
| Precision | bfloat16 autocast + channels-last on CUDA; float32 on Apple GPUs (MPS) and CPU |

All augmentation runs on the GPU inside the training loop, per batch, in this order: color jitter on
[0, 1] pixels → normalize → flip + crop → zoom (only with `--zoom`) → Cutout → CutMix. The loss is
`λ · CE(y_a) + (1 − λ) · CE(y_b)` with label smoothing inside each term. Because the training targets
are soft and mixed, the training loss and accuracy in `history.json` are not directly comparable to
the val numbers. `--mix mixup` swaps CutMix for Mixup, and `--mix-alpha` / `--mix-prob` control the
Beta parameter and the fraction of mixed batches.

The checkpoint with the best val accuracy is reloaded at the end to score the local test split exactly
once and to write the submission files.

The full recipe, as `pipeline.sbatch` and `train_pipeline.ipynb` run it (the teachers exceed the
5M limit on purpose and are only used as targets; only the 4.70M-parameter student is submitted):

```bash
# three teachers: same split (--seed 0), different training seeds
python train.py --blocks 4 4 3 --width 128 --kernel 3 --epochs 200 --patience 0 --label-smoothing 0.1 \
                --cutout 16 --mix cutmix --randaugment 2 9 --teacher-over-limit --out runs/teacher_w128
python train.py ... --train-seed 1 --out runs/teacher_w128_s1     # and --train-seed 2 -> runs/teacher_w128_s2
# the student, 4,697,162 params, distilled from the ensemble
python train.py --blocks 4 4 3 --width 64 --kernel 3 --epochs 300 --patience 0 --label-smoothing 0.1 \
                --cutout 16 --mix cutmix --randaugment 1 9 --grayscale 0.05 --kd-alpha 0.9 --kd-temp 4 \
                --teacher runs/teacher_w128/best.pt runs/teacher_w128_s1/best.pt runs/teacher_w128_s2/best.pt \
                --out runs/r24_kd_t3_e300
python model_summary.py --run runs/r24_kd_t3_e300     # parameter count, layer table, final test accuracy
# the submission: the same four runs again on all 50,000 images (--val-frac 0 --test-frac 0), last epoch kept
```

## Results

The recipe above, seed 0, one H100 (Slurm job 21775660; `runs/r24_kd_t3_e300_full/slurm_21775660.txt`
is the log of every stage). Val and local test are the 5,000-image splits; the all-50k runs have no
local score by construction, so the held-out student is their paired estimate.

| Run | Params | Val | Local test | + flip TTA | Kaggle |
|---|---|---|---|---|---|
| Teacher w128, training seed 0 (`runs/teacher_w128`) | 18.8M | 97.22% | 97.12% | 97.38% | never submitted |
| Teacher w128, training seed 1 | 18.8M | 97.50% | 97.26% | 97.60% | never submitted |
| Teacher w128, training seed 2 | 18.8M | 97.02% | 97.42% | 97.66% | never submitted |
| Teacher ensemble, averaged logits | | 97.56% | | | never submitted |
| **Student `r24_kd_t3_e300`**, 300 epochs, 27 min | **4,697,162** | 97.24% | **97.42%** | **97.64%** | |
| **Student `r24_kd_t3_e300_full`**, all 50k, 34 min | **4,697,162** | n/a | n/a | n/a | **0.907** (`submission_tta.csv`) |
| Student `r24_kd_t3_e800`, 800 epochs, 73 min | 4,697,162 | 97.54% | 97.64% | 97.82% | |
| Student `r24_kd_t3_e800_full`, all 50k, 94 min | 4,697,162 | n/a | n/a | n/a | 0.899 |
| Student `r24_kd_t3_e300_zoom`, + `--zoom 0.7 1.4`, 28 min | 4,697,162 | 96.92% | 97.22% | 97.30% | |
| Student `r24_kd_t3_e300_zoom_full`, all 50k, 35 min | 4,697,162 | n/a | n/a | n/a | 0.904 |
| Student `r24_kd_t3_e300_b453`, `--blocks 4 5 3`, 28 min | 4,992,586 | 97.42% | 97.32% | 97.42% | |
| Student `r24_kd_t3_e300_b453_full`, all 50k, 35 min | 4,992,586 | n/a | n/a | n/a | 0.903 |
| Teacher ensemble, w128 at 300 epochs (`runs/teacher_w128_e300*`) | 18.8M | 97.74% | | | never submitted |
| Student `r24_kd_t3_e300_te300`, distilled from it | 4,697,162 | 97.44% | 97.46% | 97.58% | |
| Student `r24_kd_t3_e300_te300_full`, all 50k | 4,697,162 | n/a | n/a | n/a | 0.898 |
| Teacher ensemble, width 192, 42,206,410 params each (`runs/teacher_w192*`) | 42.2M | 97.92% | | | never submitted |
| Student `r24_kd_t3_e300_tw192`, distilled from it | 4,697,162 | 97.42% | 97.54% | 97.50% | |
| Student `r24_kd_t3_e300_tw192_full`, all 50k | 4,697,162 | n/a | n/a | n/a | 0.904 |
| Student `r24_kd_t3_e300_sam`, + `--sam 0.05`, 36 min | 4,697,162 | 97.26% | 97.48% | 97.66% | |
| Student `r24_kd_t3_e300_sam_full`, all 50k | 4,697,162 | n/a | n/a | n/a | 0.901 |

The seed-0 teacher in `runs/teacher_w128` is the training the round-3 and round-4 students distilled
from (its val accuracy is the first entry of `teacher_val_accs` in their `summary.json`). The round-2
single-teacher student `runs/r24_kd` used an earlier training of the same configuration (val 97.00%,
test 97.38%), kept in git history (commit 60d4aac).

How the rounds compare, same split and seed:

| Round | Model | Local test | Kaggle |
|---|---|---|---|
| 1 | 5×5 ResNet-20, 2,986,666 params, 100 epochs, jitter / flip / crop only | 92.74% | 0.82 |
| 2 | 3×3 ResNet-24, 4,697,162 params, + label smoothing, Cutout, CutMix, 200 epochs | 96.50% | 0.88 |
| 3 | + RandAugment, distilled from one w128 teacher, 200 epochs; submission trained on all 50k | 97.24% | 0.897 |
| 3b | the same single-teacher student for 300 epochs (`runs/r24_kd_e300*`) | 97.32% / 97.48% TTA | — |
| 4 | + three-teacher ensemble, random grayscale, 300 epochs; all 50k; flip TTA | 97.42% / 97.64% TTA | **0.907** |
| 4b | the same with 800 student epochs (`EPOCHS=800`, `runs/r24_kd_t3_e800*`) | 97.64% / 97.82% TTA | 0.899 |
| 5a | round 4 + random zoom 0.7–1.4× for the student (`--zoom 0.7 1.4`) | 97.22% / 97.30% TTA | 0.904 |
| 5b | round 4 with one more block in stage 2 (`--blocks 4 5 3`, 4,992,586 params) | 97.32% / 97.42% TTA | 0.903 |
| 6a | round 4 with the teachers trained for 300 epochs (ensemble 97.74% val) | 97.46% / 97.58% TTA | 0.898 |
| 6b | round 4 with width-192 teachers, 42.2M params each (ensemble 97.92% val) | 97.54% / 97.50% TTA | 0.904 |
| 7 | round 4 with sharpness-aware minimization for the student (`--sam 0.05`) | 97.48% / 97.66% TTA | 0.901 |

The student with flip TTA is level with the ensemble it was distilled from (97.64% vs 97.56%), so the
teachers are now the ceiling. A longer schedule was the first thing tried (distillation keeps improving
with very long schedules, Beyer et al. 2022): the 800-epoch student is the best model on every local
number, val 97.54%, test 97.64% and 97.82% with flip TTA, yet scored 0.899 on Kaggle. That is a
16-image swing on the hidden set, inside its noise (next paragraph), so the two students are
indistinguishable there and the 300-epoch run stays the submitted one.

Round 5 changed one thing at a time from the round-4 recipe:
random zoom, the augmentation suggested by the shift analysis below, and the only deeper block layout
that still fits under 5M parameters. Each moved about 90 of the 2,000 Kaggle predictions and scored
0.904 and 0.903, again inside the noise of the 0.907 entry. More parameters and scale augmentation
do not close the gap either. Round 6 attacked the ceiling instead: since the student had caught up
with its teachers, the teachers were made stronger, by training them 50% longer or by making them 2.2×
wider. The ensembles did improve, to 97.74% and 97.92% on val, but the students distilled from them
stayed level with the round-4 student on the test split and scored 0.898 and 0.904. Round 7 changed the
optimizer instead, sharpness-aware minimization (Foret et al. 2021), the one remaining method whose
published gain on CIFAR-10 ResNets exceeds the leaderboard noise: it came out level on every local
number and scored 0.901. Eight submissions of this model family now span 0.897 to 0.907, a spread
smaller than the sampling error of the hidden set and unrelated to the local numbers, so the round-4
student remains the selected submission.

Two caveats when reading the Kaggle column. The hidden set has 2,000 images, so its standard error at
90% is about 0.7 points: the steps above are real, but anything smaller is leaderboard noise, and the
score must not be used to pick between seeds. And the 7-point gap to the local split is a source
shift, not a preprocessing bug: the Kaggle images match the training set in per-channel mean, std,
brightness and saturation, but visibly come from a different source (web-photo look, 3× the grayscale
fraction, slightly sharper), much like CIFAR-10.1, whose published accuracy drop is the same size.
Regularization, distillation and TTA all narrow it; nothing in the preprocessing can.

The train accuracy in `history.json` is low by construction: it is measured against mixed, smoothed
targets on heavily augmented images, not against the clean labels. On the local test split the
remaining errors are mostly cat ↔ dog confusions, followed by cat/bird/deer.

## How to run

Three ways in, all of them ending in `train.py`: run the script directly, submit one of the two Slurm
scripts on the NYU HPC (how everything in [Results](#results) was trained), or open a notebook to look at
the runs.

### Locally: `train.py`

```bash
# the round-2 recipe (no distillation), writes runs/resnet24_cutmix/; the full pipeline is in Training recipe
python train.py --blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 \
                --label-smoothing 0.1 --cutout 16 --mix cutmix --out runs/resnet24_cutmix

# quick CPU sanity check: 1 epoch on 1,000 images
python train.py --smoke --out runs/smoke

# variants: every hyperparameter is a flag, see `python train.py -h`
python train.py --blocks 2 2 2 --width 16
python train.py --mix mixup --mix-alpha 0.4 --cutout 0          # Mixup instead of CutMix, no Cutout
python train.py --zoom 0.7 1.4                                   # random zoom per image (round-5 variant)
python train.py --sam 0.05                                       # sharpness-aware minimization (2x time per epoch)
python train.py --ema 0.9995                                     # evaluate / submit an exponential moving average of the weights

# model card for any config (before training) or any finished run
python model_summary.py --blocks 4 4 3 --width 64 --kernel 3
python model_summary.py --run runs/resnet24_cutmix

# re-score a run with flip TTA, or review its least confident predictions
python predict.py --run runs/resnet24_cutmix
python inspect_predictions.py --run runs/resnet24_cutmix --split val --n 96
```

The device is picked automatically (CUDA, then Apple MPS, then CPU) or set with `--device`. The full
teacher → student recipe as a sequence of `train.py` commands is in [Training recipe](#training-recipe);
on a single GPU it is simpler to let `pipeline.sbatch` (or `train_pipeline.ipynb`) run the stages.

Every run writes these files to `--out`:

- `submission.csv`: Kaggle predictions (`id,label`); `submission_tta.csv` the same with flip TTA
- `summary.json`: config, parameter count, best epoch, val and test accuracy, predicted class counts
- `history.json`: per-epoch train/val loss and accuracy, learning rate, epoch time
- `curves.png`: loss and accuracy plots
- `best.pt`: best checkpoint, with the normalization stats and args (not tracked in git)

### On the NYU HPC: the Slurm scripts

One-time setup on the NYU Torch cluster. The repo lives as `project_1/` inside a project directory on `/scratch`
(home has a low file-count quota). Python and PyTorch go into a Singularity overlay, the standard
NYU HPC recipe, because conda/pip trees have far too many files for `/scratch` quotas otherwise:

```bash
mkdir -p /scratch/$USER/deep_learning && cd /scratch/$USER/deep_learning
git clone <this repo> project_1
mkdir -p slurm_logs
# put the competition files in project_1/data/ (see Data above)

# blank overlay + Miniforge installer (both provided by NYU HPC)
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz . && gunzip overlay-15GB-500K.ext3.gz   # ls /share/apps/overlay-fs-ext3/ for other sizes
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# install into the overlay (read-write, only this once)
singularity exec --overlay overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
cat > /ext3/env.sh <<'SH'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
SH
source /ext3/env.sh
pip install -r project_1/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # version, False (no GPU on login node)
exit
```

Both Slurm scripts are plain `sbatch` jobs. They take their options from environment variables, passed
with `sbatch --export=ALL`, so nothing inside them needs editing. Submit from `/scratch/$USER/deep_learning`:
it becomes the job's root, and the overlay is mounted read-only so several jobs can share it. Logs land in
`slurm_logs/`, each run's outputs in its run directory under `project_1/runs/`.

**`train.sbatch`: one run.** `ARGS` holds the `train.py` flags and `OUT` the run directory. Without
`OUT` the job writes to `runs/YYYYMMDD_HHMMSS/`, named by its start time:

```bash
sbatch project_1/train.sbatch                      # defaults: the 5x5 ResNet-20 baseline, runs/YYYYMMDD_HHMMSS/
OUT=project_1/runs/resnet24_cutmix \
ARGS="--blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 --label-smoothing 0.1 --cutout 16 --mix cutmix" \
sbatch --export=ALL project_1/train.sbatch
```

**`pipeline.sbatch`: the whole teacher-ensemble pipeline as one job.** Three teachers, the distilled
student, then the same four on all 50k, stages back to back on one GPU: about 2.5 hours on an H100 with
300 student epochs, and about one hour (only the two student stages) once the teachers exist. Its
configuration block hard-codes the same flags and run names as `train_pipeline.ipynb`, and it skips any
stage whose run directory already has a `summary.json`, so after a crash or time-out you submit it again
and it resumes at the first unfinished stage. This is the job that produced the submitted model:

```bash
sbatch project_1/pipeline.sbatch                                       # the real thing, 4 h limit
SMOKE=1 sbatch --export=ALL --time=00:20:00 project_1/pipeline.sbatch   # 1-epoch plumbing check, runs/smoke_pipeline/
# a student variant once the teachers exist: adds a suffix to the two student run names
SUFFIX=zoom STUDENT_EXTRA="--zoom 0.7 1.4" sbatch --export=ALL project_1/pipeline.sbatch
SUFFIX=b453 STUDENT_EXTRA="--blocks 4 5 3" sbatch --export=ALL project_1/pipeline.sbatch
# new teachers: longer or wider (new TEACHER_NAME so the existing ones are not reused)
TEACHER_EPOCHS=300 TEACHER_NAME=teacher_w128_e300 SUFFIX=te300 sbatch --export=ALL --time=07:00:00 project_1/pipeline.sbatch
TEACHER_WIDTH=192 TEACHER_NAME=teacher_w192 SUFFIX=tw192 sbatch --export=ALL --time=07:00:00 project_1/pipeline.sbatch
```

`pipeline.sbatch` has a 4-hour limit, enough for the whole pipeline on an H100 (a w128 teacher takes 17
to 31 minutes on the held-out split and 20 to 38 on all 50k; the 300-epoch student 27 and 34).
`train.sbatch` has a 3-hour limit for a single run. Override either with `sbatch --time=HH:MM:SS`. Both
use the course allocation (`--account`); check yours with
`sacctmgr show assoc user=$USER format=account%40,partition%20` and pass a different one on the `sbatch`
command line if needed. `ROOT`, `REPO`, `OVERLAY`, `SIF` and `ENV_SH` override the project directory, repo
folder name, container and activation script; `PY=/path/to/python` skips the container and runs that
interpreter directly. `pipeline.sbatch` also takes `EPOCHS`, `N_TEACHERS`, `TEACHER_EPOCHS`, `TEACHER_WIDTH`,
`TEACHER_NAME`, `SUFFIX`, `STUDENT_EXTRA` and `SMOKE`; the header comment of each script lists everything.
The two original 5×5 ResNet-20 runs predate the start-time naming and have no recorded start time:
`runs/20261029_225523/` (Slurm job 21361893, 30 epochs) and `runs/20261029_225524/` (job 21368912, 100
epochs) carry the files' creation time on disk, in job order, not the real training time.

### The notebooks: inspecting runs

Both notebooks are for analysis. They read `summary.json`, `history.json` and `best.pt` from `runs/`,
plot, and call `model_summary.report(...)` so the parameter count and final test accuracy appear in the
executed notebook. Each can also launch `train.py` (as a subprocess, streaming its log into the cell),
which is convenient from an interactive GPU session, but the tracked runs were all produced by the Slurm
scripts above. Open them from the repo root, with a kernel that has `requirements.txt` installed
(`python -m ipykernel install --user --name deep_learning` registers the environment as a kernel); on the
NYU HPC use OOD / Jupyter on a GPU node.

- `train_notebook.ipynb`: one run. `CONFIG` keys are `train.py` flag names and `RUN_NAME` the run
  directory; set `SKIP_TRAINING = True` to analyse an existing run without training. Plots the training
  curves, the confusion matrix and per-class accuracy on the local test split, the hardest mistakes, the
  predicted class distribution on the Kaggle images, and a comparison across every finished run in `runs/`.
- `train_pipeline.ipynb`: the pipeline. Its configuration cell mirrors the configuration block of
  `pipeline.sbatch` (same flags, same run names, same skip-if-finished logic), so its results cells read
  the runs the Slurm job produced: the results table of every stage, the model card of the held-out
  student, and the val curves of the student and its teachers.

## References

All code in this repository is written from scratch for the project; no pretrained weights or
external datasets are used. The techniques come from:

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016. (ResNet, BasicBlock)
- C. Szegedy et al. *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. (label smoothing)
- T. DeVries, G. W. Taylor. *Improved Regularization of Convolutional Neural Networks with Cutout.* arXiv:1708.04552, 2017.
- H. Zhang, M. Cissé, Y. N. Dauphin, D. Lopez-Paz. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018.
- S. Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019.
- I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017. (cosine schedule)
- E. D. Cubuk, B. Zoph, J. Shlens, Q. V. Le. *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space.* CVPR Workshops 2020.
- G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015. (distillation loss)
- L. Beyer, X. Zhai, A. Royer, L. Markeeva, R. Anil, A. Kolesnikov. *Knowledge Distillation: A Good Teacher Is Patient and Consistent.* CVPR 2022. (consistent teaching on the same augmented batch, long schedules)
- P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. *Sharpness-Aware Minimization for Efficiently Improving Generalization.* ICLR 2021. (`--sam`)
