#!/bin/bash
# A6000 template — DO NOT EDIT IN PLACE (R-Slurm.3).
# Submit via:
#   sbatch --export=CMD="scripts/train_finetune.py --config configs/exp/<name>/config.yaml" \
#          slurm/templates/a6000.slurm
# Or copy this file to slurm/exp_<name>.slurm for an experiment-specific tweak.

#SBATCH --job-name=iseg
#SBATCH --account=accre_guests_acc
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --partition=batch_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=outputs/slurm/%j.out

set -euo pipefail

# Env stack — A6000 (x86_64, CUDA 12.6).
setup_accre_software_stack
module load cuda/12.6
export LD_PRELOAD=/usr/lib64/libnvidia-ml.so.1

# Shared venv (R-Env.2). Do NOT create a per-project venv under /home/lih30/ (no quota).
source /home/lih30/allinone/bin/activate

# Working directory (code repo only — checkout under /home/lih30/ is small).
cd /home/lih30/interactive_seg

# Sanity: outputs/ must be a symlink to /data/h_oguz_lab/lih30/interactive_seg_outputs/
# (R-Data.AccreHome — /home has no quota for checkpoints/logs).
if [ ! -L outputs ]; then
  echo "ERROR: outputs/ must be a symlink to /data/h_oguz_lab/lih30/interactive_seg_outputs/" >&2
  exit 1
fi

# Print job info to stdout for traceability.
echo "Job ${SLURM_JOB_ID} on $(hostname) — GPU: $(nvidia-smi -L | head -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Command: ${CMD:-scripts/train_finetune.py --config configs/exp/smoke.yaml}"

python ${CMD:-scripts/train_finetune.py --config configs/exp/smoke.yaml}
