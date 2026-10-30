# Small ResNets under 5M parameters: Kaggle CIFAR-10 challenge

A CIFAR-style ResNet for a Kaggle CIFAR-10 ResNet architecture challenge whose one hard rule is
**at most 5 million trainable parameters**, trained from scratch. Depth, width and kernel size are
flags. The first model used **5×5 kernels everywhere** (ResNet-20, 3.0M params); the current recipe is
a **3×3 ResNet-24 with stage widths 64/128/256 (4.70M params)** trained for 200 epochs with label
smoothing, Cutout and CutMix. The training pipeline is a single PyTorch script that keeps the whole
dataset on the GPU and runs augmentation there too, so a 100-epoch ResNet-20 run takes about
6.5 minutes on one A100.

**Best result so far:** **88% on the Kaggle hidden test set** with the 4.70M-parameter 3×3 ResNet-24
recipe, up from 82% with the 5×5 ResNet-20 baseline (`runs/baseline_21368912`, which scored 93.44% val
and 92.74% on the local held-out split). See [Recent changes](#recent-changes) for what moved the score.

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
└─ stem: conv k×k(3→w) · BN · ReLU
   ├─ stage 1: n1 × BasicBlock(w),   32×32
   ├─ stage 2: n2 × BasicBlock(2w),  16×16   (first block stride 2)
   └─ stage 3: n3 × BasicBlock(4w),   8×8    (first block stride 2)
      └─ global average pool · linear(4w → 10)
```

Each `BasicBlock` is conv k×k · BN · ReLU · conv k×k · BN, plus a shortcut, then ReLU (the residual
block from the project handout). The shortcut is the identity, or a 1×1 conv + BN projection when the
stride or channel count changes. Depth is `2·sum(blocks) + 2`. Blocks per stage (`--blocks`), base
width (`--width`) and kernel size (`--kernel`) are flags; `train.py` exits if the result has more than
5M parameters. Parameter counts, from `model_summary.py`:

| Config | Flags | Params | Under 5M |
|---|---|---|---|
| 5×5 ResNet-20 (baseline, the defaults) | `--blocks 3 3 3 --width 32 --kernel 5` | 2,986,666 | yes |
| 5×5 ResNet-20, wider | `--blocks 3 3 3 --width 40 --kernel 5` | 4,664,530 | yes |
| **3×3 ResNet-24, widths 64/128/256 (current recipe)** | `--blocks 4 4 3 --width 64 --kernel 3` | **4,697,162** | yes |
| 5×5 ResNet-24, width 40 | `--blocks 4 4 3 --width 40 --kernel 5` | 5,065,010 | no |

The 3×3 ResNet-24 is ResNet-18 with its last (512-channel) stage removed and one extra block in each
of the first two stages: that spends the budget on depth and width at 32×32 and 16×16 resolution
instead of on a single 512-channel stage.

## Training recipe

| | |
|---|---|
| Optimizer | SGD, Nesterov momentum 0.9, lr 0.1, batch 128 |
| Weight decay | 5e-4 on conv/linear weights only (none on BN or biases) |
| Schedule | 1 epoch of linear warmup, then cosine decay to 0 |
| Augmentation (GPU) | color jitter (brightness/contrast/saturation ±0.2), random horizontal flip, 4-px zero-pad random crop |
| Normalization | per-channel mean/std computed on the training split |
| Precision | bfloat16 autocast + channels-last on CUDA; float32 on Apple GPUs (MPS) and CPU |
| Epochs | 100, early stopping after 20 epochs without a new best val accuracy |

The checkpoint with the best val accuracy is reloaded at the end to score the local test split and
write `submission.csv`.

### Regularization flags (off by default, on in the current recipe)

The baseline reaches 99.97% train accuracy while val loss creeps back up, so the second round adds
regularization. All of it runs on the GPU inside the training loop; the flags default to off so the
tracked baseline runs stay reproducible.

| Flag | Recipe value | What it does |
|---|---|---|
| `--label-smoothing` | 0.1 | cross-entropy against soft targets (Szegedy et al., 2016) |
| `--cutout` | 16 | zeroes one random 16×16 square per image after normalization (DeVries & Taylor, 2017) |
| `--mix` | `cutmix` | `cutmix` pastes a random box from a shuffled partner image and mixes the labels by area (Yun et al., 2019); `mixup` blends whole images (Zhang et al., 2018) |
| `--mix-alpha` | 1.0 | mixing weight λ ~ Beta(α, α), one λ per batch |
| `--mix-prob` | 1.0 | fraction of batches that are mixed |
| `--epochs 200 --patience 0` | | run the full cosine schedule; with CutMix the best epoch is near the end |

Per-batch order: color jitter on [0, 1] pixels → normalize → flip + crop → Cutout → Mixup / CutMix.
With label smoothing or mixing on, the training loss and accuracy in `history.json` are measured
against the soft / mixed targets, so they are not directly comparable to the val numbers.

The full current recipe:

```bash
python train.py --blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 \
                --label-smoothing 0.1 --cutout 16 --mix cutmix --out runs/resnet24_cutmix
python model_summary.py --run runs/resnet24_cutmix     # parameter count, layer table, final test accuracy
```

## Results

| Run | Epochs | Color jitter | Best epoch | Val acc | Local test acc | Train time (A100) |
|---|---|---|---|---|---|---|
| `baseline_21361893` | 30 | off | 29 | 91.62% | 92.14% | 75 s |
| `baseline_21368912` | 100 | 0.2 / 0.2 / 0.2 | 90 | **93.44%** | **92.74%** | 390 s |

Training curves for the 100-epoch run:

![training curves](runs/baseline_21368912/curves.png)

Training accuracy reaches 99.97% by the end, while val loss bottoms out and then creeps back up. That
observation led to the regularization round described in the next section.

### Kaggle hidden test set

| Model | Params | Recipe | Kaggle score |
|---|---|---|---|
| 5×5 ResNet-20 (`baseline_21368912`) | 2,986,666 | 100 epochs, jitter + flip + crop | 0.82 |
| 3×3 ResNet-24, widths 64/128/256 | 4,697,162 | 200 epochs, + label smoothing, Cutout, CutMix | **0.88** |

The local held-out split is an optimistic estimate: the baseline scored 92.74% locally but 82% on
Kaggle, so the hidden images are drawn from a harder or shifted distribution. Changes that improve
generalization therefore show up more on Kaggle than on the local split. The run directory, val and
local test accuracy of the ResNet-24 run will be added to the table above when it is committed.

## Recent changes

Second round of the project: the Kaggle score went from **0.82 to 0.88**. Five things changed; all are
flags in `train.py`, and the exact command is in [Training recipe](#training-recipe).

1. **A bigger, better-shaped network** (`--blocks 4 4 3 --width 64 --kernel 3`). The baseline used 5×5
   filters and only 3.0M of the 5M-parameter budget. The new model uses standard 3×3 filters, is one
   stage deeper per early block, and is twice as wide at the 32×32 and 16×16 resolutions, for 4.70M
   parameters. Two stacked 3×3 convolutions cover the same area as one 5×5 with fewer weights and an
   extra nonlinearity, so the same budget buys more useful capacity, and the wider early stages give
   more detectors for the fine textures that separate cat from dog or truck from automobile.

2. **Label smoothing** (`--label-smoothing 0.1`). Instead of training the network to be 100% certain
   of the correct class, it is trained to be 90% certain with the rest spread over the other classes.
   This stops it from becoming overconfident on training images, which makes its predictions better
   calibrated and more accurate on images it has never seen.

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

Why the gain is larger on Kaggle than locally: every item after the first is a regularizer. They
penalize memorizing the training set and reward features that transfer, which is exactly what a
shifted hidden test set rewards.

What did not change: SGD with Nesterov momentum, learning rate 0.1, weight decay 5e-4, batch 128, one
warmup epoch then cosine decay, the 40k/5k/5k split, color jitter, and flip-and-crop were all already
in the baseline. The new flags default to off, so the baseline runs in the table above still reproduce.
Alongside the recipe, `model_summary.py` was added to print the layer table, the trainable-parameter
count against the 5M limit, and a run's final test accuracy, and `train.py` now refuses to train a
model over the limit.

## Usage

```bash
# full run: 100 epochs with early stopping, writes runs/baseline/
python train.py

# quick CPU sanity check: 1 epoch on 1,000 images
python train.py --smoke

# other variants
python train.py --blocks 2 2 2 --width 16
python train.py --epochs 200 --patience 0 --jitter 0 0 0 --out runs/e200
python train.py --mix mixup --mix-alpha 0.4 --cutout 0          # Mixup instead of CutMix, no Cutout

# model card for any config (before training) or any finished run
python model_summary.py --blocks 4 4 3 --width 64 --kernel 3
python model_summary.py --run runs/baseline_21368912
```

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
sbatch kaggle_resnet_cifar10/train.sbatch
OUT=kaggle_resnet_cifar10/runs/e200 ARGS="--epochs 200 --patience 0" sbatch --export=ALL kaggle_resnet_cifar10/train.sbatch
# current recipe (3x3 ResNet-24, 4.70M params, 200 epochs, label smoothing + Cutout + CutMix)
OUT=kaggle_resnet_cifar10/runs/resnet24_cutmix \
ARGS="--blocks 4 4 3 --width 64 --kernel 3 --epochs 200 --patience 0 --label-smoothing 0.1 --cutout 16 --mix cutmix" \
sbatch --export=ALL --time=01:30:00 kaggle_resnet_cifar10/train.sbatch
```

The wider 3×3 model does roughly twice the work per epoch of the 5×5 ResNet-20, so a 200-epoch run
needs more than the job file's default 45 minutes; the `--time` override above covers it.

The job file uses the course allocation (`--account`, `--partition`); pass different ones on the
`sbatch` command line if needed. `PY`, `OVERLAY`, `SIF`, `OUT` and `ARGS` environment variables
override the Python used, wrap it in a Singularity container, and set the run directory and flags.
Logs land in `slurm_logs/`; each run's outputs land in `OUT`.

Every run writes these files to `--out`:

- `submission.csv`: Kaggle predictions (`id,label`)
- `summary.json`: config, parameter count, best epoch, val and test accuracy, predicted class counts
- `history.json`: per-epoch train/val loss and accuracy, learning rate, epoch time
- `curves.png`: loss and accuracy plots
- `best.pt`: best checkpoint, with the normalization stats and args (not tracked in git)

Requirements: Python 3.9+ (3.12 on the cluster), PyTorch, NumPy, Pillow, and matplotlib.
`torchinfo` is optional: if installed, `model_summary.py` prints its parameter count as a cross-check.

### Running on a Mac (Apple GPU)

On an Apple-silicon Mac the script uses the GPU through PyTorch's MPS backend automatically; no CUDA
is needed. The stock PyTorch wheel includes MPS support:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy pillow matplotlib
python train.py --smoke --out runs/smoke   # quick check; prints "device mps"
python train.py                            # full run on the Apple GPU
```

The device is picked in the order CUDA, MPS, CPU; override with `--device cpu`. MPS runs in
float32 (no bfloat16 autocast). Measured on an M3 Pro: about 41 s per epoch at batch 128, so a
100-epoch run takes roughly 70 minutes versus 6.5 minutes on an A100. Use the cluster for sweeps
and the Mac for development and short runs (`--epochs 30` finishes in about 20 minutes).

## References

All code in this repository is written from scratch for the project; no pretrained weights or
external datasets are used. The techniques come from:

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016. (ResNet, BasicBlock)
- C. Szegedy et al. *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. (label smoothing)
- T. DeVries, G. W. Taylor. *Improved Regularization of Convolutional Neural Networks with Cutout.* arXiv:1708.04552, 2017.
- H. Zhang, M. Cissé, Y. N. Dauphin, D. Lopez-Paz. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018.
- S. Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019.
- I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017. (cosine schedule)
