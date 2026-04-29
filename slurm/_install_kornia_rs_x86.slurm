#!/bin/bash
#SBATCH --job-name=iseg_kr_x86
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=outputs/slurm/%j.out
set -e
source /home/lih30/allinone/bin/activate
cd /home/lih30/interactive_seg
pip install --no-deps kornia_rs
python -c "import kornia; print(\"kornia\", kornia.__version__, \"OK\")"
echo DONE
