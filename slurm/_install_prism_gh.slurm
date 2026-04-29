#!/bin/bash
#SBATCH --job-name=iseg_prism_gh
#SBATCH --account=h_oguz_lab_acc
#SBATCH --gres=gpu:nvidia_gh200:1
#SBATCH --partition=batch_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%j.out

set -e
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
export HOME=/home/lih30/aarch64
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
source "$HOME/GH/bin/activate"
cd /home/lih30/interactive_seg

echo "=== current torch on GH ==="
python -c "import torch; print(torch.__version__)"

echo "=== install missing PRISM deps via uv pip --no-deps ==="
uv pip install --no-deps kornia torchio prefetch-generator MedPy

echo "=== verify ==="
python -c "import torch, kornia, torchio, prefetch_generator, medpy; print('torch', torch.__version__); print('kornia', kornia.__version__); print('torchio', torchio.__version__); print('PRISM GH deps OK')"
echo DONE
