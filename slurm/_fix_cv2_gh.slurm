#!/bin/bash
#SBATCH --job-name=iseg_cv2_fix
#SBATCH --account=h_oguz_lab_acc
#SBATCH --gres=gpu:nvidia_gh200:1
#SBATCH --partition=batch_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm/%j.out
set -e
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
export HOME=/home/lih30/aarch64
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
source "$HOME/GH/bin/activate"
echo "=== current cv2 ==="
python -c "import cv2; print('cv2', cv2.__version__); print('ximgproc:', hasattr(cv2, 'ximgproc'))" || true
echo
echo "=== uninstall opencv-python + opencv-python-headless ==="
uv pip uninstall opencv-python opencv-python-headless || true
echo
echo "=== install opencv-contrib-python (--no-deps) ==="
uv pip install --no-deps opencv-contrib-python
echo
echo "=== verify ==="
python -c "import cv2; print('cv2', cv2.__version__); print('ximgproc.thinning:', hasattr(cv2.ximgproc, 'thinning'))"
echo DONE
