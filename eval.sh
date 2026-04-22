#!/bin/bash

# ============================================================
# LeWM evaluation job on Oscar (Brown CCV)
#
# Usage:
#   sbatch eval.sh
#   sbatch eval.sh lewm_epoch_28
#   sbatch eval.sh lewm_epoch_20 /users/$USER/scratch/le-wm-me/lewm-pusht
#   sbatch eval.sh lewm_epoch_28 /users/$USER/scratch/le-wm-me/lewm-pusht eval.num_eval=10
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J lewm_eval
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Positional args:
#   1) policy checkpoint name, without "_object.ckpt"
#   2) cache_dir containing both checkpoint(s) and dataset .h5
POLICY_NAME=${1:-lewm_epoch_28}
CACHE_DIR=${2:-/users/$USER/scratch/le-wm-me/lewm-pusht}
shift 2 || true
EXTRA_ARGS="$@"

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Policy:    $POLICY_NAME"
echo "Cache dir: $CACHE_DIR"
echo "Extra:     $EXTRA_ARGS"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

# --- environment -------------------------------------------------
source /oscar/scratch/$USER/.venv/bin/activate

echo "python: $(which python)  ($(python -V 2>&1))"
python -c "import stable_worldmodel, torch; print('stable_worldmodel OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

# --- sanity checks ----------------------------------------------
DATA_FILE="$CACHE_DIR/pusht_expert_train.h5"
CKPT_FILE="$CACHE_DIR/${POLICY_NAME}_object.ckpt"

if [ ! -e "$DATA_FILE" ]; then
    echo "ERROR: missing dataset file: $DATA_FILE"
    exit 1
fi

if [ ! -e "$CKPT_FILE" ]; then
    echo "ERROR: missing checkpoint file: $CKPT_FILE"
    echo "Hint: pass policy name without suffix, e.g. lewm_epoch_28"
    exit 1
fi

echo "dataset:    $DATA_FILE ($(du -h "$DATA_FILE" | cut -f1))"
echo "checkpoint: $CKPT_FILE ($(du -h "$CKPT_FILE" | cut -f1))"

# --- eval --------------------------------------------------------
srun python eval.py --config-name=pusht.yaml \
    policy="$POLICY_NAME" \
    cache_dir="$CACHE_DIR" \
    $EXTRA_ARGS

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
