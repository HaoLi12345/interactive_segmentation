#!/bin/bash
#SBATCH --job-name=iseg_test
#SBATCH --account=h_oguz_lab_acc
#SBATCH --gres=gpu:nvidia_gh200:1
#SBATCH --partition=batch_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%j.out
set -e
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
export HOME=/home/lih30/aarch64
export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
source "$HOME/GH/bin/activate"
cd /home/lih30/interactive_seg
export PYTHONPATH=/home/lih30/interactive_seg:/home/lih30/interactive_seg/src:$PYTHONPATH
echo "GPU:" $(nvidia-smi -L)
python src/test.py \
  --data colon \
  --data_dir /data/h_oguz_lab/lih30/interactive_seg_data/msd_colon \
  --save_dir /data/h_oguz_lab/lih30/interactive_seg_outputs \
  --save_name MICCAI2024 \
  --checkpoint best \
  --multiple_outputs --refine --refine_test \
  --use_box --use_scribble --num_clicks 10 \
  --iter_nums 11 \
  --num_clicks_validation 10 \
  --image_size 128 \
  --batch_size 1 --num_workers 0
echo TEST_DONE
