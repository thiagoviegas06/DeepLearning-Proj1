#!/bin/bash
#SBATCH --job-name=resnet_pipeline
#SBATCH --account=torch_pr_495_tandon_advanced
#SBATCH --gres=gpu:h100:1          # GPU type: h100 (also h200), a100 as fallback; avoid rtx6000. `sinfo -o "%P %G"` lists them
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/rl3725/deep_learning/slurm_logs/resnet_pipeline_%j.out
#SBATCH --error=/scratch/rl3725/deep_learning/slurm_logs/resnet_pipeline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rl3725@nyu.edu
#
# The whole teacher-ensemble -> student pipeline of train_pipeline.ipynb as ONE Slurm job, stages back
# to back on one GPU (same flags, same run names, so the notebook's results cells read these runs):
#
#   1. N_TEACHERS w128 teachers on the held-out split   runs/teacher_w128, _s1, _s2   ~31 min each
#   2. the distilled student                             runs/r24_kd_t3_e300           ~75 min
#   3. the same teachers on all 50k                      runs/teacher_w128_full, ...   ~37 min each
#   4. the submission student on all 50k                 runs/r24_kd_t3_e300_full      ~95 min
#
# About 6.5 h in total for 3 teachers. A stage whose run directory already has summary.json is
# skipped, so if the job dies or hits the time limit just submit it again and it carries on from the
# first unfinished stage. Resubmit after any edit to EPOCHS/N_TEACHERS: the run names carry them.
#
#   mkdir -p slurm_logs && sbatch project_1/pipeline.sbatch
#   SUFFIX=zoom STUDENT_EXTRA="--zoom 0.7 1.4" sbatch --export=ALL project_1/pipeline.sbatch   # student variant -> runs/r24_kd_t3_e300_zoom*
#   SMOKE=1 sbatch --export=ALL --time=00:20:00 project_1/pipeline.sbatch   # 1 epoch, 1k images, runs/smoke_pipeline/
#
# Same environment plumbing as train.sbatch: overlay + /ext3/env.sh, or PY=/path/to/python to skip the
# container. Override ROOT, REPO, SIF, OVERLAY, ENV_SH, PY, EPOCHS, N_TEACHERS, TEACHER_EPOCHS, SMOKE
# through the environment with sbatch --export=ALL.
set -euo pipefail
ROOT="${ROOT:-/scratch/rl3725/deep_learning}"
REPO="${REPO:-project_1}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-$ROOT/overlay-15GB-500K.ext3}"
ENV_SH="${ENV_SH:-/ext3/env.sh}"

# ---- configuration: keep in sync with the "Configuration" cell of train_pipeline.ipynb ----
EPOCHS="${EPOCHS:-300}"                 # student epochs
N_TEACHERS="${N_TEACHERS:-3}"           # ensemble size; teacher k trains with --train-seed k
TEACHER_EPOCHS="${TEACHER_EPOCHS:-200}"
TEACHER_NAME="${TEACHER_NAME:-teacher_w128}"
SEED=0                                  # the data split, shared by every run
GRAYSCALE=0.05                          # student only
SMOKE="${SMOKE:-}"                      # non-empty: --smoke on every stage, into runs/smoke_pipeline/
SUFFIX="${SUFFIX:-}"                    # run-name suffix for a student variant (the teachers are reused as they are)
STUDENT_EXTRA="${STUDENT_EXTRA:-}"      # extra student flags, e.g. "--zoom 0.7 1.4" or "--blocks 4 5 3" (a repeated flag overrides BASE: last one wins)
TAG="t${N_TEACHERS}_e${EPOCHS}${SUFFIX:+_$SUFFIX}"

BASE="--blocks 4 4 3 --kernel 3 --patience 0 --batch 128 --lr 0.1 --wd 5e-4 --warmup 1.0 \
--jitter 0.2 0.2 0.2 --label-smoothing 0.1 --cutout 16 --mix cutmix --seed $SEED"
TEACHER="--width 128 --epochs $TEACHER_EPOCHS --randaugment 2 9 --teacher-over-limit"   # 18,765,962 params
STUDENT="--width 64 --epochs $EPOCHS --randaugment 1 9 --kd-alpha 0.9 --kd-temp 4 --grayscale $GRAYSCALE"  # 4,697,162
FULL="--val-frac 0 --test-frac 0"       # all 50k; best.pt = last epoch
# --------------------------------------------------------------------------------------------

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd "$ROOT"
RUNS="$REPO/runs"; [[ -n "$SMOKE" ]] && RUNS="$RUNS/smoke_pipeline"
echo "host $(hostname)  $(date)  job ${SLURM_JOB_ID:-?}  restart ${SLURM_RESTART_COUNT:-0}  runs $RUNS  tag $TAG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# python ARGS...: train.py through the chosen interpreter (see train.sbatch)
python() {
    if [[ -n "${PY:-}" ]]; then
        "$PY" -u "$@"
    else
        singularity exec --nv --overlay "$OVERLAY:ro" "$SIF" \
            /bin/bash -c 'source "$0" && exec python -u "$@"' "$ENV_SH" "$@"
    fi
}

# run_stage NAME FLAGS...: one train.py run into $RUNS/NAME, skipped when it already has summary.json
run_stage() {
    local name="$1"; shift
    local out="$RUNS/$name"
    if [[ -f "$out/summary.json" ]]; then
        echo "[$name] already done, skipping ($out/summary.json)"; return
    fi
    local t0; t0=$(date +%s)
    echo "[$name] $(date)  train.py --out $out $* ${SMOKE:+--smoke}"
    mkdir -p "$out"
    # shellcheck disable=SC2086  # the flag strings are meant to split
    local rc=0
    python "$REPO/train.py" --out "$out" "$@" ${SMOKE:+--smoke} 2>&1 | tee "$out/log.txt" || rc="${PIPESTATUS[0]}"
    echo "[$name] exit $rc  ($(( ($(date +%s) - t0) / 60 )) min wall)"
    [[ "$rc" -eq 0 ]] || exit "$rc"
}

# teacher k: name, flags and best.pt path (k = 0 is plain TEACHER_NAME, the existing run)
teachers=(); teachers_full=(); teacher_pts=""; teacher_full_pts=""
for ((k = 0; k < N_TEACHERS; k++)); do
    name="$TEACHER_NAME"; [[ $k -gt 0 ]] && name="${TEACHER_NAME}_s$k"
    teachers+=("$name"); teachers_full+=("${name}_full")
    teacher_pts+=" $RUNS/$name/best.pt"; teacher_full_pts+=" $RUNS/${name}_full/best.pt"
done
seed_flag() { [[ "$1" -gt 0 ]] && echo "--train-seed $1"; return 0; }

# shellcheck disable=SC2086
{
    for ((k = 0; k < N_TEACHERS; k++)); do
        run_stage "${teachers[$k]}"      $BASE $TEACHER $(seed_flag "$k")
    done
    run_stage "r24_kd_$TAG"              $BASE $STUDENT $STUDENT_EXTRA --teacher $teacher_pts
    for ((k = 0; k < N_TEACHERS; k++)); do
        run_stage "${teachers_full[$k]}" $BASE $TEACHER $FULL $(seed_flag "$k")
    done
    run_stage "r24_kd_${TAG}_full"       $BASE $STUDENT $STUDENT_EXTRA $FULL --teacher $teacher_full_pts
}
echo "done $(date)   upload: $RUNS/r24_kd_${TAG}_full/submission_tta.csv (or submission.csv without flip TTA)"

