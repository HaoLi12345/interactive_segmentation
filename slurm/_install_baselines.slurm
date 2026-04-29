#!/bin/bash
#SBATCH --job-name=iseg_baselines
#SBATCH --account=h_oguz_lab_acc
#SBATCH --gres=gpu:nvidia_gh200:1
#SBATCH --partition=batch_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%j.out

set -e
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
export HOME=/home/lih30/aarch64
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
source "$HOME/GH/bin/activate"
cd /home/lih30/interactive_seg

echo "=== install nninteractive (--no-deps) ==="
uv pip install --no-deps nninteractive
echo "=== install nninteractive transitive deps explicitly (avoid torch upgrade) ==="
# nnunetv2 is required at runtime; install --no-deps to avoid torch tug-of-war
uv pip install --no-deps nnunetv2 acvl-utils dynamic_network_architectures dicom2nifti numpyencoder

echo "=== verify nnInteractive import on GH ==="
python -c "from nnInteractive.inference.inference_session import nnInteractiveInferenceSession; print('nnInteractive import OK')"

echo "=== download VISTA3D weights (~872MB) -> /data/h_oguz_lab/lih30/foundation_models/vista3d ==="
mkdir -p /data/h_oguz_lab/lih30/foundation_models
python <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="MONAI/vista3d", local_dir="/data/h_oguz_lab/lih30/foundation_models/vista3d")
print("vista3d at:", p)
PY

echo "=== download nnInteractive weights (~400MB) -> /data/h_oguz_lab/lih30/foundation_models/nninteractive ==="
python <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="nnInteractive/nnInteractive",
                       allow_patterns=["nnInteractive_v1.0/*"],
                       local_dir="/data/h_oguz_lab/lih30/foundation_models/nninteractive")
print("nnInteractive at:", p)
PY

echo "=== final sizes ==="
du -sh /data/h_oguz_lab/lih30/foundation_models/*
echo DONE
