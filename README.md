# ResNet-24 under 5M parameters: Kaggle CIFAR-10 challenge

A CIFAR-style ResNet for a Kaggle CIFAR-10 architecture challenge whose one hard rule is **at most
5 million trainable parameters**, trained from scratch on the provided data only. The model is a
**3×3 ResNet-24 with stage widths 64/128/256 (4,697,162 parameters)** trained for 200 epochs with SGD,
a cosine schedule, label smoothing, Cutout and CutMix. The whole pipeline is one PyTorch script that
keeps the dataset on the GPU and runs augmentation there too, so a full 200-epoch run takes about
13 minutes on one A100.

**Result:** 96.66% validation accuracy, 96.50% on the local held-out split, and **0.88 on the Kaggle
hidden test set**. See [Results](#results) and [Recent changes](#recent-changes).

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
train.sbatch          SLURM job script (NYU HPC, A100, Singularity container)
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
| Epochs | 200, no early stopping (`--patience 0`), so the full cosine schedule runs |
| Augmentation (GPU) | color jitter (brightness/contrast/saturation ±0.2), random horizontal flip, 4-px zero-pad random crop |
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
once and to write `submission.csv`.

The full recipe:

```bash
python train.py --blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 \
                --label-smoothing 0.1 --cutout 16 --mix cutmix --out runs/resnet24_cutmix
python model_summary.py --run runs/resnet24_cutmix     # parameter count, layer table, final test accuracy
```

## Results

ResNet-24, 4,697,162 parameters, the recipe above, seed 0, one A100 (also the executed run in
`train_notebook.ipynb`):

| Metric | Value |
|---|---|
| Epochs run / best epoch | 200 / 194 |
| Train accuracy, last epoch (against CutMix + smoothed targets) | 72.6% |
| Val accuracy (best epoch) | **96.66%** |
| Local held-out test accuracy (scored once with `best.pt`) | **96.50%** |
| Kaggle hidden test set | **0.88** |
| Training time | 766 s (3.8 s / epoch) |

The train accuracy is low by construction: it is measured against mixed, smoothed targets on heavily
augmented images, not against the clean labels. On the local test split the remaining errors are
mostly cat ↔ dog confusions, followed by cat/bird/deer. The Kaggle hidden set scores lower than the
local split, so the local number is an optimistic estimate.

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

## Usage

```bash
# current recipe, writes runs/resnet24_cutmix/
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

- `submission.csv`: Kaggle predictions (`id,label`)
- `summary.json`: config, parameter count, best epoch, val and test accuracy, predicted class counts
- `history.json`: per-epoch train/val loss and accuracy, learning rate, epoch time
- `curves.png`: loss and accuracy plots
- `best.pt`: best checkpoint, with the normalization stats and args (not tracked in git)

Requirements: Python 3.9+ (3.12 on the cluster), PyTorch, NumPy, Pillow, and matplotlib.
`torchinfo` is optional: if installed, `model_summary.py` prints its parameter count as a cross-check.

### NYU HPC

One-time setup on the cluster. The repo lives as `kaggle_resnet_cifar10/` inside a project
directory on `/scratch` (home has a low file-count quota, so the venv goes on scratch too):

```bash
mkdir -p /scratch/$USER/deep_learning && cd /scratch/$USER/deep_learning
git clone <this repo> kaggle_resnet_cifar10
python3 -m venv .venv && .venv/bin/pip install torch numpy pillow matplotlib
mkdir -p slurm_logs
# put the competition files in kaggle_resnet_cifar10/data/ (see Data above)
```

Submit from that directory; it becomes the job's root, and the venv in it is used:

```bash
OUT=kaggle_resnet_cifar10/runs/resnet24_cutmix \
ARGS="--blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 --label-smoothing 0.1 --cutout 16 --mix cutmix" \
sbatch --export=ALL --time=01:30:00 kaggle_resnet_cifar10/train.sbatch
```

The job file's default time limit is 45 minutes, so the 200-epoch run needs the `--time` override
above. It uses the course allocation (`--account`, `--partition`); pass different ones on the `sbatch`
command line if needed. `PY`, `OVERLAY`, `SIF`, `OUT` and `ARGS` environment variables override the
Python used, wrap it in a Singularity container, and set the run directory and flags. Logs land in
`slurm_logs/`; each run's outputs land in `OUT`. When `OUT` is not set, the job writes to
`runs/YYYYMMDD_HHMMSS/`, named by the job's start time. The two original 5×5 ResNet-20 runs predate
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
