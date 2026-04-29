#!/bin/bash
# GH200 template — DO NOT EDIT IN PLACE (R-Slurm.3).
# Submit via:
#   sbatch --export=CMD="scripts/train_finetune.py --config configs/exp/<name>/config.yaml" \
#          slurm/templates/gh200.slurm

#SBATCH --job-name=iseg
#SBATCH --account=h_oguz_lab_acc
#SBATCH --gres=gpu:nvidia_gh200:1
#SBATCH --partition=batch_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=outputs/slurm/%j.out

set -euo pipefail

# GH200 is aarch64 — must use the ARM venv, not the x86 allinone (R-Env.3).
# Triton cache must be per-job to avoid cross-job collisions.
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}

source /home/lih30/aarch64/GH/bin/activate

cd /home/lih30/interactive_seg

# Sanity: outputs/ must be a symlink to /data/h_oguz_lab/lih30/interactive_seg_outputs/
if [ ! -L outputs ]; then
  echo "ERROR: outputs/ must be a symlink to /data/h_oguz_lab/lih30/interactive_seg_outputs/" >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID} on $(hostname) — GPU: $(nvidia-smi -L | head -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Command: ${CMD:-scripts/train_finetune.py --config configs/exp/smoke.yaml}"

python ${CMD:-scripts/train_finetune.py --config configs/exp/smoke.yaml}
