#!/bin/bash
#SBATCH --job-name=iseg_smoke
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
cd /home/lih30/interactive_seg && export PYTHONPATH=/home/lih30/interactive_seg:/home/lih30/interactive_seg/src:$PYTHONPATH

echo "GPU:" $(nvidia-smi -L)

# Background GPU memory poller: write timestamp + memory.used MiB every 2s
MEMLOG="outputs/slurm/${SLURM_JOB_ID}_gpu_mem.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 2 > "$MEMLOG" 2>&1 &
GPUWATCH_PID=$!

python src/train.py \
  --data colon \
  --data_dir /data/h_oguz_lab/lih30/interactive_seg_data/msd_colon \
  --save_dir /data/h_oguz_lab/lih30/interactive_seg_outputs \
  --save_name smoke \
  --max_epoch 1 \
  --multiple_outputs \
  --refine \
  --image_size 128 \
  --batch_size 2 --num_workers 0 --use_box --dynamic

# Stop poller and report peak
kill $GPUWATCH_PID 2>/dev/null || true
sleep 1
echo "=== peak GPU memory used during run ==="
awk -F',' 'NR>1 {gsub(/[^0-9]/,"",$2); if ($2+0 > max) max=$2+0} END {print "peak: " max " MiB"}' "$MEMLOG"
echo "(full timeseries: $MEMLOG)"
echo SMOKE_DONE
