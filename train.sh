#!/bin/bash

# ============================================================
# LeWM training job on Oscar (Brown CCV)
#
# Usage:
#   sbatch train.sh                 # defaults to data=pusht
#   sbatch train.sh pusht           # same as above
#   sbatch train.sh pusht trainer.max_epochs=5 loader.batch_size=64
#                                   # extra args are forwarded to Hydra
#
# Monitor:
#   myq                             # job status
#   tail -f slurm-<jobid>.out       # stdout
#   tail -f slurm-<jobid>.err       # stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J lewm_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# First positional arg is the Hydra data config (default: pusht).
# Any remaining args are forwarded to Hydra as overrides.
DATA=${1:-pusht}
shift || true
EXTRA_ARGS="$@"

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Data:      $DATA"
echo "Extra:     $EXTRA_ARGS"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

# --- environment -------------------------------------------------
# Activate your venv (change the path if yours differs).
source /oscar/scratch/$USER/.venv/bin/activate

# Checkpoint output dir used by train.py (swm.data.utils.get_cache_dir()).
export STABLEWM_HOME=/oscar/scratch/$USER/stablewm_home
mkdir -p "$STABLEWM_HOME"

# DataLoader stability.
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Uncomment to run wandb offline (e.g. when debugging on a login node).
# export WANDB_MODE=offline

# --- train -------------------------------------------------------
srun python train.py data="$DATA" $EXTRA_ARGS

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
