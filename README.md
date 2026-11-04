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
images. See [Results](#results) and [Recent changes](#recent-changes).

## Competition rules this code enforces

| Rule | Where |
|---|---|
| ≤ 5,000,000 trainable parameters | `train.py` refuses to start a larger model; `model_summary.py` prints the count and the margin |
| Train from scratch on the provided CIFAR-10 only | no pretrained weights or external data anywhere; `best.pt` is written by this script |
| Clearly print the parameter count and final test accuracy | `python model_summary.py --run runs/<name>`, also called from the notebook |

## Repository layout

```
train.py              model, data pipeline, augmentation, training loop, Kaggle submission writer
model_summary.py      layer table, trainable-parameter count vs the 5M limit, final test accuracy of a run
train_notebook.ipynb  drives train.py, then plots curves, confusion matrix, per-class accuracy, Kaggle counts
train_pipeline.ipynb  the teacher-ensemble -> distilled-student pipeline, held-out split then all 50k
train.sbatch          SLURM job script: one train.py run (NYU HPC, A100, Singularity container)
pipeline.sbatch       SLURM job script: the whole teacher-ensemble -> student pipeline as one job
runs/<name>/          output of each run: summary.json, history.json, curves.png, submission.csv
data/                 competition data (not tracked, see below)
```

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
| Distillation | three teachers, same architecture at width 128 (18,765,962 params, `--teacher-over-limit`), same data split, training seeds 0/1/2; the student matches their averaged logits with the Hinton KL loss at temperature 4, weight 0.9 (`--kd-alpha 0.9 --kd-temp 4`) |
| Test-time augmentation | predictions averaged over the image and its mirror (`submission_tta.csv`) |
| Cutout | one random 16×16 square per image zeroed after normalization (`--cutout 16`) |
| CutMix | a random box from a shuffled partner image is pasted in and the labels are mixed by area, λ ~ Beta(1, 1), every batch (`--mix cutmix`) |
| Label smoothing | 0.1 (`--label-smoothing 0.1`) |
| Normalization | per-channel mean/std computed on the training split |
| Precision | bfloat16 autocast + channels-last on CUDA; float32 on Apple GPUs (MPS) and CPU |

All augmentation runs on the GPU inside the training loop, per batch, in this order: color jitter on
[0, 1] pixels → normalize → flip + crop → Cutout → CutMix. The loss is
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
| Teacher w128, training seed 0 (`runs/teacher_w128`) | 18.8M | 97.00% | 97.38% | 97.48% | never submitted |
| Teacher w128, training seed 1 | 18.8M | 97.50% | 97.26% | 97.60% | never submitted |
| Teacher w128, training seed 2 | 18.8M | 97.02% | 97.42% | 97.66% | never submitted |
| Teacher ensemble, averaged logits | | 97.56% | | | never submitted |
| **Student `r24_kd_t3_e300`**, 300 epochs, 27 min | **4,697,162** | 97.24% | **97.42%** | **97.64%** | |
| **Student `r24_kd_t3_e300_full`**, all 50k, 34 min | **4,697,162** | n/a | n/a | n/a | **0.907** (`submission_tta.csv`) |

How the rounds compare, same split and seed:

| Round | Model | Local test | Kaggle |
|---|---|---|---|
| 1 | 5×5 ResNet-20, 2,986,666 params, 100 epochs, jitter / flip / crop only | 92.74% | 0.82 |
| 2 | 3×3 ResNet-24, 4,697,162 params, + label smoothing, Cutout, CutMix, 200 epochs | 96.50% | 0.88 |
| 3 | + RandAugment, distilled from one w128 teacher, 200 epochs; submission trained on all 50k | 97.24% | 0.897 |
| 4 | + three-teacher ensemble, random grayscale, 300 epochs; all 50k; flip TTA | 97.42% / 97.64% TTA | **0.907** |
| 4b | the same with 800 student epochs (`EPOCHS=800`, not in `runs/`) | 97.64% / 97.82% TTA | 0.899 |

The student with flip TTA is level with the ensemble it was distilled from (97.64% vs 97.56%), so the
teachers are now the ceiling. A longer schedule was the first thing tried (distillation keeps improving
with very long schedules, Beyer et al. 2022): the 800-epoch student is the best model on every local
number, val 97.54%, test 97.64% and 97.82% with flip TTA, yet scored 0.899 on Kaggle. That is a
16-image swing on the hidden set, inside its noise (next paragraph), so the two students are
indistinguishable there and the 300-epoch run stays the submitted one.

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

## Recent changes

Second round of the project. The first version was a 5×5 ResNet-20 (`--blocks 3 3 3 --width 32
--kernel 5`, 2,986,666 params) trained for 100 epochs with early stopping and only color jitter, flip
and crop as augmentation; it scored 0.82 on Kaggle. Five things changed, all of them flags in
`train.py`, and the exact command is in [Training recipe](#training-recipe).

1. **A bigger, better-shaped network** (`--blocks 4 4 3 --width 64 --kernel 3`). The previous model
   used 5×5 filters and only 3.0M of the 5M-parameter budget. The new one uses standard 3×3 filters,
   has one more block in each of the first two stages, and is twice as wide at the 32×32 and 16×16
   resolutions, for 4.70M parameters. Two stacked 3×3 convolutions cover the same area as one 5×5
   with fewer weights and an extra nonlinearity, so the same budget buys more useful capacity, and the
   wider early stages give more detectors for the fine textures that separate cat from dog or truck
   from automobile.

2. **Label smoothing** (`--label-smoothing 0.1`). Instead of training the network to be 100% certain
   of the correct class, it is trained to be 90% certain with the rest spread over the other classes.
   This stops it from becoming overconfident on training images and gives better-calibrated
   predictions on images it has never seen.

3. **Cutout** (`--cutout 16`). A random 16×16 square of every training image is blanked out. The
   network can no longer rely on a single telltale spot, such as a wheel or a beak, and has to use the
   whole object, so partially occluded or oddly framed test images hurt less.

4. **CutMix** (`--mix cutmix`). A random patch from another training image is pasted in and the label
   is mixed in proportion to the pasted area, for example 70% dog and 30% ship. This manufactures
   endless new training images from the same 40,000, forces the network to localize objects rather
   than memorize backgrounds, and again teaches softer, better-calibrated predictions.

5. **Longer training, no early stopping** (`--epochs 200 --patience 0`). With this much augmentation
   the training task is harder, so the model needs more epochs, and running the full cosine schedule
   lets the learning rate decay smoothly to zero. With CutMix the best epoch is almost always near the
   end, so early stopping was turned off.

The motivation was that the previous model reached 99.97% train accuracy while its val loss crept
back up, and it scored 92.74% on the local held-out split but only 0.82 on Kaggle. Everything after
item 1 is a regularizer: it penalizes memorizing the training set and rewards features that transfer,
which is what a shifted hidden test set rewards. The gain is therefore larger on Kaggle than locally.

What did not change: SGD with Nesterov momentum, learning rate 0.1, weight decay 5e-4, batch 128, one
warmup epoch then cosine decay, the 40k/5k/5k split, color jitter, and flip-and-crop. The new flags
default to off, so the earlier runs in `runs/` still reproduce. Alongside the recipe, `model_summary.py`
was added to print the layer table, the trainable-parameter count against the 5M limit and a run's
final test accuracy, `train.py` now refuses to train a model over the limit, and `train_notebook.ipynb`
was added to drive the script and inspect a run.

### Third and fourth rounds: distillation, RandAugment, all 50k

The round-2 model was 96.5% locally and 0.88 on Kaggle; the local split was close to saturated, so
the next steps aimed at the shifted hidden set. Again every item is a `train.py` flag, all off by
default. Runs with these flags are the `r24_*` and `teacher_*` directories in `runs/`.

1. **RandAugment** (`--randaugment N M`, Cubuk et al. 2020): N random ops per image out of rotate,
   shear, translate, brightness, contrast, color, sharpness, posterize, solarize, equalize and
   autocontrast, at magnitude M, implemented on the GPU like the rest of the pipeline. One op at
   magnitude 9 gave +0.5 points locally on its own (`runs/r24_ra1_9`); two ops suit the bigger teachers.

2. **Knowledge distillation** (`--teacher RUN/best.pt ...`, Hinton et al. 2015). A width-128 copy of the
   same network (18.8M parameters, allowed only as a teacher with `--teacher-over-limit`, never
   submitted) is trained first and reaches about 97.4% locally. The 4.70M student then sees the same
   augmented, mixed batches and matches the teacher's temperature-4 softened predictions through a KL
   term with weight 0.9, keeping 0.1 of the ordinary smoothed cross-entropy. The student caught up with
   its single teacher (97.24%), so round 4 trains **three** teachers that differ only in `--train-seed`
   on the same split and distils from their **averaged logits**: the ensemble is 97.56% on val, and
   the student with flip TTA now matches it (97.64%).

3. **Random grayscale** for the student (`--grayscale 0.05`), the one measurable difference between
   the Kaggle images and the training set (3.3% grayscale images vs 1.1%).

4. **Longer student schedule** (`--epochs 300`): with the distillation signal the student keeps
   improving late in the cosine schedule (best epoch 289 of 300 on the held-out split).

5. **Training on all 50,000 images** for the submission (`--val-frac 0 --test-frac 0`). Every stage is
   run twice: once on the 40k/5k/5k split, which gives the honest local numbers above, and once on all
   50k, where `best.pt` is simply the last epoch and the local score is `null`. The two runs differ
   only in the data split, so the held-out run's accuracy is reported next to the submission.

6. **Test-time augmentation** at scoring time only: `submission_tta.csv` averages the logits of each
   image and its mirror (+0.2 points locally, 52 of 2,000 Kaggle predictions changed). Both local
   accuracies land in `summary.json`.

The ensemble-then-student pipeline is `train_pipeline.ipynb` for Jupyter and `pipeline.sbatch` for
Slurm; both skip any stage that already has a `summary.json`, so a crashed job resumes. Each round's
Kaggle score is in the table under [Results](#results).

## Usage

```bash
# the round-2 recipe (no distillation), writes runs/resnet24_cutmix/; the full pipeline is in Training recipe
python train.py --blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 \
                --label-smoothing 0.1 --cutout 16 --mix cutmix --out runs/resnet24_cutmix

# quick CPU sanity check: 1 epoch on 1,000 images
python train.py --smoke --out runs/smoke

# variants: every hyperparameter is a flag, see `python train.py -h`
python train.py --blocks 2 2 2 --width 16
python train.py --mix mixup --mix-alpha 0.4 --cutout 0          # Mixup instead of CutMix, no Cutout

# model card for any config (before training) or any finished run
python model_summary.py --blocks 4 4 3 --width 64 --kernel 3
python model_summary.py --run runs/resnet24_cutmix
```

`train_notebook.ipynb` runs the same recipe through `train.py` (its `CONFIG` keys are the flag names),
then plots the curves, the confusion matrix and per-class accuracy on the local test split, and the
predicted class distribution on the Kaggle images. It also calls `model_summary.report(...)` so the
parameter count and final test accuracy are printed in the executed notebook.

Every run writes these files to `--out`:

- `submission.csv`: Kaggle predictions (`id,label`); `submission_tta.csv` the same with flip TTA
- `summary.json`: config, parameter count, best epoch, val and test accuracy, predicted class counts
- `history.json`: per-epoch train/val loss and accuracy, learning rate, epoch time
- `curves.png`: loss and accuracy plots
- `best.pt`: best checkpoint, with the normalization stats and args (not tracked in git)

Requirements: Python 3.9+ (3.12 on the cluster), PyTorch, NumPy, Pillow, and matplotlib.
`torchinfo` is optional: if installed, `model_summary.py` prints its parameter count as a cross-check.

### NYU HPC

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
pip install torch numpy pillow matplotlib torchinfo
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # version, False (no GPU on login node)
exit
```

Submit from `/scratch/$USER/deep_learning`; it becomes the job's root and the overlay is mounted
read-only, so several jobs can share it:

```bash
OUT=project_1/runs/resnet24_cutmix \
ARGS="--blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 --label-smoothing 0.1 --cutout 16 --mix cutmix" \
sbatch --export=ALL project_1/train.sbatch
```

To run the whole teacher-ensemble pipeline of `train_pipeline.ipynb` as one job instead (three
teachers, the distilled student, then the same four on all 50k, stages back to back on one GPU,
about 2.5 hours on an H100 with 300 student epochs, and only the two student stages, about one
hour, once the teachers exist), use `pipeline.sbatch`. It mirrors the notebook's configuration and run names, and
skips any stage whose run directory already has a `summary.json`, so after a crash or time-out you
just submit it again and it resumes at the first unfinished stage:

```bash
sbatch project_1/pipeline.sbatch                                       # the real thing, 7 h limit
SMOKE=1 sbatch --export=ALL --time=00:20:00 project_1/pipeline.sbatch   # 1-epoch plumbing check
```

The job file's default time limit is 3 hours, enough for every stage of the teacher-ensemble pipeline
in `train_pipeline.ipynb` (a w128 teacher takes about 37 minutes on all 50k; the 3-teacher, 300-epoch
distilled student about 95). Override with `sbatch --time=HH:MM:SS` for longer runs. It uses the course allocation (`--account`, `--partition`); check yours with
`sacctmgr show assoc user=$USER format=account%40,partition%20` and pass different ones on the
`sbatch` command line if needed. `ROOT`, `REPO`, `OVERLAY`, `SIF`, `ENV_SH`, `OUT` and `ARGS` environment
variables override the project directory, repo folder name, container, activation script, run
directory and flags; `PY=/path/to/python` skips the container and runs that interpreter directly.
Logs land in `slurm_logs/`; each run's outputs land in `OUT`. When `OUT` is not set, the job writes
to `runs/YYYYMMDD_HHMMSS/`, named by the job's start time. The two original 5×5 ResNet-20 runs predate
this scheme and have no recorded start time: `runs/20261029_225523/` (Slurm job 21361893, 30 epochs)
and `runs/20261029_225524/` (job 21368912, 100 epochs) carry the files' creation time on disk, in job
order, not the real training time.


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
