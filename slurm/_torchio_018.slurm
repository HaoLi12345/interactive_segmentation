#!/bin/bash
#SBATCH --job-name=iseg_tio018
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=outputs/slurm/%j.out
set -e
source /home/lih30/allinone/bin/activate
echo "before:" $(python -c "import torch,torchio; print(torch.__version__, torchio.__version__)")
pip install "torchio==0.18.91"
echo "after:" $(python -c "import torch,torchio; print(torch.__version__, torchio.__version__)")
echo DONE
