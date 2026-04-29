#!/bin/bash
#SBATCH --job-name=iseg_kr_gh
#SBATCH --account=h_oguz_lab_acc
#SBATCH --gres=gpu:nvidia_gh200:1
#SBATCH --partition=batch_gpu
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=outputs/slurm/%j.out
set -e
export TRITON_CACHE_DIR=/tmp/triton_
export HOME=/home/lih30/aarch64
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
source "$HOME/GH/bin/activate"
cd /home/lih30/interactive_seg
uv pip install --no-deps kornia_rs
python -c "import kornia; print(\"kornia\", kornia.__version__, \"OK\")"
echo DONE
