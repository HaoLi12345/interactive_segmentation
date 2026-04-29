#!/bin/bash
# CPU-only template — DO NOT EDIT IN PLACE (R-Slurm.3).
# Use for: data sanity checks, prepare_data, eval CSV aggregation, etc.
# Submit via:
#   sbatch --export=CMD="scripts/sanity_check_dataset.py --dataset kits21" slurm/templates/cpu.slurm

#SBATCH --job-name=iseg_cpu
# No --account needed for the default (h_oguz_lab) on the CPU "batch" partition.
# The "_acc" suffix accounts (accre_guests_acc / h_oguz_lab_acc) are only valid for batch_gpu.
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%j.out

# Use `set -e` only; `set -u` triggers unbound-variable errors in lmod.sh.
set -e

# CPU jobs don't need the CUDA module stack; just activate the shared x86 venv.
source /home/lih30/allinone/bin/activate

cd /home/lih30/interactive_seg

if [ ! -L outputs ]; then
  echo "ERROR: outputs/ must be a symlink to /data/h_oguz_lab/lih30/interactive_seg_outputs/" >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Command: ${CMD}"

python ${CMD}
