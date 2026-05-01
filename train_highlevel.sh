#!/bin/bash

# ============================================================
# HWM (high-level world model + macro action encoder) on Push-T
#
# Requires a trained LeWM checkpoint (ModelObjectCallBack pickle):
#   <name>_epoch_<N>_object.ckpt
#
# Usage:
#   sbatch train_highlevel.sh /path/to/lewm_epoch_15_object.ckpt
#   sbatch train_highlevel.sh $STABLEWM_HOME/<jobid>/lewm_epoch_15_object.ckpt trainer.max_epochs=200
#   # extra args are forwarded to Hydra (after low_level_ckpt)
#
# Monitor:
#   myq
#   tail -f slurm-<jobid>.out
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -J hwm_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: sbatch train_highlevel.sh <low_level_ckpt> [hydra overrides...]"
  echo "Example:"
  echo "  sbatch train_highlevel.sh \"\$STABLEWM_HOME/<run_id>/lewm_epoch_15_object.ckpt\""
  exit 1
fi

LOW_LEVEL_CKPT="$1"
shift || true
EXTRA_ARGS="$@"

echo "============================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "low_level_ckpt:  $LOW_LEVEL_CKPT"
echo "Extra:            $EXTRA_ARGS"
echo "Node:             $(hostname)"
echo "Started:          $(date)"
echo "GPU:              $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

if [ ! -f "$LOW_LEVEL_CKPT" ]; then
  echo "ERROR: low_level_ckpt not found: $LOW_LEVEL_CKPT"
  exit 1
fi

# --- environment (match train.sh) --------------------------------
source /oscar/scratch/$USER/.venv/bin/activate

echo "python: $(which python)  ($(python -V 2>&1))"
python -c "import sys; print('sys.executable =', sys.executable)"
python -c "import datasets, os; print('datasets', datasets.__version__, 'at', os.path.dirname(datasets.__file__))"
python -c "from datasets import config as _; print('datasets.config OK')" \
  || { echo 'ERROR: HuggingFace datasets broken.'; exit 1; }
python -c "import stable_pretraining; import torch; \
  print('stable_pretraining OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

# --- data root (same HDF5 as flat LeWM: pusht_expert_train.h5) -----
: "${STABLEWM_HOME:=/oscar/scratch/$USER/stablewm_home}"
export STABLEWM_HOME
mkdir -p "$STABLEWM_HOME"
echo "STABLEWM_HOME = $STABLEWM_HOME"

DATA_FILE="$STABLEWM_HOME/pusht_expert_train.h5"
REPO_PUSHT="$SLURM_SUBMIT_DIR/lewm-pusht/pusht_expert_train.h5"
if [ ! -e "$DATA_FILE" ] && [ -e "$REPO_PUSHT" ]; then
  echo "Linking pusht HDF5 from repo into STABLEWM_HOME..."
  ln -sf "$REPO_PUSHT" "$DATA_FILE"
fi
if [ ! -e "$DATA_FILE" ]; then
  echo "ERROR: missing $DATA_FILE (needed for data=pusht_waypoints)"
  echo "  ln -sf /path/to/pusht_expert_train.h5 $DATA_FILE"
  exit 1
fi
echo "data: $DATA_FILE ($(du -h "$DATA_FILE" | cut -f1))"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export WANDB_MODE=${WANDB_MODE:-online}
echo "WANDB_MODE = $WANDB_MODE"

# config/train/hwm.yaml defaults include data: pusht_waypoints; set explicitly for clarity.
srun python train_highlevel.py \
  low_level_ckpt="$LOW_LEVEL_CKPT" \
  data=pusht_waypoints \
  $EXTRA_ARGS

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
