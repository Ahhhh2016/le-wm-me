#!/bin/bash

# ============================================================
# HWM hierarchical eval (pusht_hwm + HierarchicalCEMSolver).
# Push-T Tab. 10 recipe is d_l = 50 macro latent; solver reads d_l from the
# high checkpoint unless you override solver.d_l=...
#
# Usage (Oscar):
#   sbatch eval_hwm.sh
#   sbatch eval_hwm.sh eval.num_eval=10
#
# Local:
#   bash eval_hwm.sh
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J hwm_eval
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# --- checkpoints (names relative to each cache root, without _object.ckpt) ---
CACHE_DATA="/users/yliu674/scratch/le-wm-me/lewm-pusht"
CACHE_HIGH="/users/yliu674/scratch/stablewm_home"

POLICY_LOW="lewm_epoch_10"
POLICY_HIGH="fresh_exp_2/hwm_epoch_100"

EXTRA_ARGS="$@"

echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Low:       ${CACHE_DATA}/${POLICY_LOW}_object.ckpt"
echo "High:      ${CACHE_HIGH}/${POLICY_HIGH}_object.ckpt"
echo "Data dir:  ${CACHE_DATA}"
echo "Extra:     ${EXTRA_ARGS}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# --- environment (Oscar CCV; adjust if needed) ---
if [ -f "/oscar/scratch/${USER}/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "/oscar/scratch/${USER}/.venv/bin/activate"
fi

echo "python: $(which python)  ($(python -V 2>&1))"
python -c "import stable_worldmodel, torch; print('stable_worldmodel OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

# --- sanity checks ---
DATA_FILE="${CACHE_DATA}/pusht_expert_train.h5"
LOW_CKPT="${CACHE_DATA}/${POLICY_LOW}_object.ckpt"
HIGH_CKPT="${CACHE_HIGH}/${POLICY_HIGH}_object.ckpt"

if [ ! -e "$DATA_FILE" ]; then
  echo "ERROR: missing dataset file: $DATA_FILE"
  exit 1
fi
if [ ! -e "$LOW_CKPT" ]; then
  echo "ERROR: missing low checkpoint: $LOW_CKPT"
  exit 1
fi
if [ ! -e "$HIGH_CKPT" ]; then
  echo "ERROR: missing high checkpoint: $HIGH_CKPT"
  exit 1
fi

echo "dataset:    $DATA_FILE ($(du -h "$DATA_FILE" | cut -f1))"
echo "low ckpt:   $LOW_CKPT ($(du -h "$LOW_CKPT" | cut -f1))"
echo "high ckpt:  $HIGH_CKPT ($(du -h "$HIGH_CKPT" | cut -f1))"

# --- eval (Push-T d_l=50 Tab. 10: macro latent dim is read from the HWM ckpt) ---
if command -v srun >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
  LAUNCH=(srun)
else
  LAUNCH=()
fi

"${LAUNCH[@]}" python eval.py --config-name=pusht_hwm \
  policy="${POLICY_LOW}" \
  policy_high="${POLICY_HIGH}" \
  cache_dir="${CACHE_DATA}" \
  policy_cache_dir="${CACHE_DATA}" \
  policy_high_cache_dir="${CACHE_HIGH}" \
  ${EXTRA_ARGS}

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
