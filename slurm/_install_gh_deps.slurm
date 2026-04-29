#!/bin/bash
#SBATCH --job-name=iseg_gh_install
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

# Pattern from /home/lih30/aarch64/install_endo.sh — must export HOME and add ~/.local/bin to PATH
# so uv (installed there) is on PATH.
export HOME=/home/lih30/aarch64
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
source "$HOME/GH/bin/activate"

cd /home/lih30/interactive_seg

echo "=== which python / uv ==="
which python
which uv

echo "=== current torch ==="
python -c "import torch; print('torch', torch.__version__)"

echo "=== uv pip install monai cc3d surface-distance segment-anything pycocotools (--no-deps) ==="
uv pip install --no-deps monai connected-components-3d surface-distance segment-anything pycocotools -v

echo "=== verifying imports ==="
python -c "import torch, monai, cc3d, surface_distance; print('torch', torch.__version__); print('monai', monai.__version__)"
python -c "from segment_anything import sam_model_registry; print('segment-anything OK')"
echo DONE
