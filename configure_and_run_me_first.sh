#!/usr/bin/env bash
# Foldername: analysis_transformer_model_curiosity_project
# Filename: configure_and_run_me_first.sh
# Local: bash configure_and_run_me_first.sh local; Slurm: bash configure_and_run_me_first.sh submit
set -euo pipefail

# The same script runs on the login node and as each Slurm array worker.
RUN_MODE=${1:-local}
if [[ "$RUN_MODE" == --help || "$RUN_MODE" == -h ]]; then
    echo 'Usage: bash configure_and_run_me_first.sh [local|submit]'
    echo 'local: train the 30 LOSO folds for d64_l1_h4, seed 42.'
    echo 'submit: submit cache and training arrays to Slurm.'
    echo 'Set PYTHON_BIN, BOLD_TS_ROOT, BEH_DIR; optionally DEVICE, EPOCHS, CHECK_ONLY=1.'
    exit 0
fi
case "$RUN_MODE" in
    submit | local)
        SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        ;;
    --cache | --train)
        if [[ -n ${PROJECT_DIR:-} ]]; then
            SCRIPT_DIR=$PROJECT_DIR
        elif [[ -n ${SLURM_SUBMIT_DIR:-} ]]; then
            SCRIPT_DIR=$SLURM_SUBMIT_DIR
        else
            SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        fi
        ;;
    *)
        echo "Usage: bash configure_and_run_me_first.sh [--cache|--train]" >&2
        exit 1
        ;;
esac
[[ $(basename "$SCRIPT_DIR") == analysis_transformer_model_curiosity_project ]] || {
    echo "Wrong project directory: $SCRIPT_DIR" >&2
    exit 1
}
export PROJECT_DIR="$SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Figure 5 model settings and input paths.
export T_BOLD=18
export MERGE_ROIS=1
export LABEL_MODE=subj # C subjective; loader fixes T to objective FLevel.
export MODEL_ARCH=d64_l1_h4
export TRAIN_SEED=42
export RESULTS_DIR="$SCRIPT_DIR/results_arch/$MODEL_ARCH/seed$TRAIN_SEED"
TRAIN_CONCURRENCY=30
export EPOCHS=${EPOCHS:-80}
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export BOLD_TS_ROOT=${BOLD_TS_ROOT:-"$SCRIPT_DIR/bold_signals_standard"}
export BEH_DIR=${BEH_DIR:-"$SCRIPT_DIR/beh_fmri"}
export SMALLVOLUME_CACHE_DIR=${SMALLVOLUME_CACHE_DIR:-"$SCRIPT_DIR/cache_voxels"}

activate_python() {
    if [[ -n ${PYTHON_BIN:-} ]]; then
        [[ -x "$PYTHON_BIN" ]] || {
            echo "PYTHON_BIN not executable: $PYTHON_BIN" >&2
            return 1
        }
        return
    fi
    PYTHON_BIN=$(command -v python)
    export PYTHON_BIN
}
activate_python

# Build or reuse the voxel cache for one participant.
if [[ "$RUN_MODE" == --cache ]]; then
    TASK_INDEX=${SLURM_ARRAY_TASK_ID:?Submit through bash configure_and_run_me_first.sh}
    SUBJECT=$(printf 'par%02d' "$((TASK_INDEX + 1))")
    "$PYTHON_BIN" -u analysis_transformer_train.py --cache-subject "$SUBJECT"
    echo "Voxel cache ready: $SUBJECT"
    exit 0
fi

# One held-out participant per training task.
if [[ "$RUN_MODE" == --train ]]; then
    TASK_INDEX=${SLURM_ARRAY_TASK_ID:?Submit via bash configure_and_run_me_first.sh}
    ((TASK_INDEX >= 0 && TASK_INDEX < 30)) || {
        echo 'array index out of range' >&2
        exit 1
    }
    SUBJECT=$(printf 'par%02d' "$((TASK_INDEX + 1))")
    "$PYTHON_BIN" -u analysis_transformer_train.py --held-out "$SUBJECT" --device "${DEVICE:-cpu}" --epochs "$EPOCHS"
    exit 0
fi

# Check inputs before submitting jobs. Training starts after all cache jobs succeed.
"$PYTHON_BIN" -u analysis_transformer_train.py --verify-inputs
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
    exit 0
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "Training: 30 LOSO folds; model=$MODEL_ARCH; seed=$TRAIN_SEED; epochs=$EPOCHS."
    exit 0
fi
if [[ "$RUN_MODE" == local ]]; then
    "$PYTHON_BIN" -u analysis_transformer_train.py --all --device "${DEVICE:-cpu}" --epochs "$EPOCHS"
    exit 0
fi

command -v sbatch >/dev/null || {
    echo 'sbatch unavailable' >&2
    exit 1
}
mkdir -p logs "$SMALLVOLUME_CACHE_DIR" results_arch
SBATCH_COMMON=(--nodes=1 --ntasks=1 --cpus-per-task=8 --chdir="$SCRIPT_DIR")
if [[ -n ${SLURM_ACCOUNT:-} ]]; then SBATCH_COMMON+=(--account="$SLURM_ACCOUNT"); fi
if [[ -n ${SLURM_PARTITION:-} ]]; then SBATCH_COMMON+=(--partition="$SLURM_PARTITION"); fi
CACHE_JOB=$(sbatch --parsable "${SBATCH_COMMON[@]}" --job-name=tf_vox_cache \
    --output=logs/voxels_%A_%a.out --error=logs/voxels_%A_%a.err \
    --mem=64G --time=02:00:00 --array=0-29%30 configure_and_run_me_first.sh --cache)
CACHE_JOB=${CACHE_JOB%%;*}
sbatch "${SBATCH_COMMON[@]}" --job-name=tf_svc_vox \
    --output=logs/train_%A_%a.out --error=logs/train_%A_%a.err \
    --mem=32G --time=03:00:00 --dependency="afterok:$CACHE_JOB" \
    --array="0-29%${TRAIN_CONCURRENCY}" configure_and_run_me_first.sh --train
echo "Submitted 30 LOSO fits after voxel-cache job $CACHE_JOB. Results: $RESULTS_DIR"
