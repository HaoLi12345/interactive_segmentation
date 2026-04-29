#!/bin/bash
#SBATCH --job-name=iseg_tio_gh
#SBATCH --account=h_oguz_lab_acc
#SBATCH --gres=gpu:nvidia_gh200:1
#SBATCH --partition=batch_gpu
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=outputs/slurm/%j.out
set -e
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
export HOME=/home/lih30/aarch64
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
source "$HOME/GH/bin/activate"
uv pip install --no-deps "torchio==0.19.6"
python -c "import torchio; print(\"torchio\", torchio.__version__, \"OK\")"
echo DONE
