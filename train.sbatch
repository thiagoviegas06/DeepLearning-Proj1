#!/bin/bash
#SBATCH --job-name=resnet_cifar
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --requeue
#SBATCH --output=/scratch/tjv235/deep_learning/slurm_logs/resnet_cifar_%j.out
#SBATCH --error=/scratch/tjv235/deep_learning/slurm_logs/resnet_cifar_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tjv235@nyu.edu
#
# Train the 5x5-conv ResNet baseline (kaggle_resnet_cifar10/train.py).
#
#   sbatch kaggle_resnet_cifar10/train.sbatch
#   OUT=kaggle_resnet_cifar10/runs/e200 ARGS="--epochs 200 --patience 0" sbatch --export=ALL kaggle_resnet_cifar10/train.sbatch
#
# Writes OUT/summary.json, history.json, curves.png, best.pt and submission.csv.
set -euo pipefail
ROOT=/scratch/tjv235/deep_learning
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/tjv235/neuro3.ext3}"
OUT="${OUT:-$ROOT/kaggle_resnet_cifar10/runs/baseline_${SLURM_JOB_ID:-local}}"
ARGS="${ARGS:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd "$ROOT"
echo "host $(hostname)  $(date)  job ${SLURM_JOB_ID:-?}  restart ${SLURM_RESTART_COUNT:-0}  out $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# shellcheck disable=SC2086  # ARGS is meant to split into separate flags
singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" \
    "$ROOT/.venv/bin/python" -u kaggle_resnet_cifar10/train.py --out "$OUT" $ARGS
date
