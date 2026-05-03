#!/bin/bash

# ============================================================
# Hierarchical (HWM + LeWM) evaluation on Oscar — pusht_hwm.yaml
#
# Defaults match yliu674 scratch paths:
#   low (LeWM):  .../lewm-pusht/lewm_epoch_10_object.ckpt
#   high (HWM):  .../stablewm_home/fresh_exp_2/hwm_epoch_100_object.ckpt
#
# Usage:
#   sbatch eval_hwm.sh
#   sbatch eval_hwm.sh /path/to/hwm.ckpt /path/to/lewm.ckpt /path/to/cache_with_dataset
#   sbatch eval_hwm.sh /path/to/hwm.ckpt /path/to/lewm.ckpt /path/to/cache eval.num_eval=10
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J lewm_hwm_eval
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

DEFAULT_HIGH_CKPT="/users/yliu674/scratch/stablewm_home/fresh_exp_2/hwm_epoch_100_object.ckpt"
DEFAULT_LOW_CKPT="/users/yliu674/scratch/le-wm-me/lewm-pusht/lewm_epoch_10_object.ckpt"
DEFAULT_CACHE_DIR="/users/yliu674/scratch/le-wm-me/lewm-pusht"

HIGH_CKPT="${1:-$DEFAULT_HIGH_CKPT}"
LOW_CKPT="${2:-$DEFAULT_LOW_CKPT}"
CACHE_DIR="${3:-$DEFAULT_CACHE_DIR}"
shift 3 || true
EXTRA_ARGS="$@"

echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "HWM (high):  $HIGH_CKPT"
echo "LeWM (low):  $LOW_CKPT"
echo "Cache dir:   $CACHE_DIR"
echo "Extra:       $EXTRA_ARGS"
echo "Node:        $(hostname)"
echo "Started:     $(date)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

# --- environment -------------------------------------------------
source /oscar/scratch/$USER/.venv/bin/activate

echo "python: $(which python)  ($(python -V 2>&1))"
python -c "import stable_worldmodel, torch; print('stable_worldmodel OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

# --- sanity checks ----------------------------------------------
DATA_FILE="$CACHE_DIR/pusht_expert_train.h5"

if [ ! -e "$DATA_FILE" ]; then
    echo "ERROR: missing dataset file: $DATA_FILE"
    exit 1
fi

if [ ! -e "$LOW_CKPT" ]; then
    echo "ERROR: missing low-level checkpoint: $LOW_CKPT"
    exit 1
fi

if [ ! -e "$HIGH_CKPT" ]; then
    echo "ERROR: missing high-level checkpoint: $HIGH_CKPT"
    exit 1
fi

echo "dataset:       $DATA_FILE ($(du -h "$DATA_FILE" | cut -f1))"
echo "low checkpoint:  $LOW_CKPT ($(du -h "$LOW_CKPT" | cut -f1))"
echo "high checkpoint: $HIGH_CKPT ($(du -h "$HIGH_CKPT" | cut -f1))"

# policy = LeWM (low), policy_high = HWM (high) — see eval.py
srun python eval.py --config-name=pusht_hwm \
    policy="$LOW_CKPT" \
    policy_high="$HIGH_CKPT" \
    cache_dir="$CACHE_DIR" \
    $EXTRA_ARGS

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
