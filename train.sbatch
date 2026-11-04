#!/bin/bash
#SBATCH --job-name=resnet_cifar
#SBATCH --account=torch_pr_495_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/rl3725/deep_learning/slurm_logs/resnet_cifar_%j.out
#SBATCH --error=/scratch/rl3725/deep_learning/slurm_logs/resnet_cifar_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rl3725@nyu.edu
#
# Train the CIFAR-10 ResNet (project_1/train.py) on NYU Greene.
#
# Layout expected under ROOT=/scratch/rl3725/deep_learning:
#   project_1/                 this repo (train.py, data/, runs/)
#   overlay-15GB-500K.ext3     Singularity overlay with Miniforge + torch installed in /ext3
#                              and /ext3/env.sh (one-time setup, see README "NYU HPC")
#   slurm_logs/                must exist before sbatch (Slurm opens the log files itself)
#
#   mkdir -p slurm_logs && sbatch project_1/train.sbatch
#   OUT=project_1/runs/e200 ARGS="--epochs 200 --patience 0" sbatch --export=ALL project_1/train.sbatch
#
# Measured on one A100 (runs/*/summary.json): ResNet-24 200 epochs 13 min, 300 epochs 18 min, w128
# teacher 200 epochs 31 min (37 on all 50k), student with 1 teacher 23 min (26 on all 50k). The
# 3-teacher 300-epoch student is ~75 min (~95 on all 50k), hence the 3 h default; override with
# sbatch --time=HH:MM:SS for anything longer.
#
# Without OUT the run goes to project_1/runs/YYYYMMDD_HHMMSS/ (job start time, e.g.
# runs/20261029_225524). Writes OUT/summary.json, history.json, curves.png, best.pt and submission.csv.
# Override ROOT, REPO, SIF, OVERLAY, ENV_SH, PY, OUT, ARGS through the environment (sbatch --export=ALL).
set -euo pipefail
ROOT="${ROOT:-/scratch/rl3725/deep_learning}"
REPO="${REPO:-project_1}"
SIF="${SIF:-/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-$ROOT/overlay-15GB-500K.ext3}"
STAMP="$(date +%Y%m%d_%H%M%S)"   # job start time; the Slurm job id is echoed below for cross-reference
OUT="${OUT:-$ROOT/$REPO/runs/$STAMP}"
ARGS="${ARGS:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd "$ROOT"
echo "host $(hostname)  $(date)  job ${SLURM_JOB_ID:-?}  restart ${SLURM_RESTART_COUNT:-0}  out $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Two ways to find Python:
#   PY=/path/to/python   run it directly (a conda env on /scratch or $HOME, no container)
#   otherwise            singularity + overlay (read-only, so several jobs can share it), and
#                        ENV_SH (default /ext3/env.sh) puts the env's python on PATH.
# "$@" keeps OUT intact even if it contains spaces.
# shellcheck disable=SC2086  # ARGS is meant to split into separate flags
if [[ -n "${PY:-}" ]]; then
    "$PY" -u "$REPO/train.py" --out "$OUT" $ARGS
else
    singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" \
        /bin/bash -c 'source "$0" && exec python -u "$@"' "${ENV_SH:-/ext3/env.sh}" \
        "$REPO/train.py" --out "$OUT" $ARGS
fi
date
