#!/bin/bash
#SBATCH --job-name=iseg_full_tio
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=outputs/slurm/%j.out
set -e
source /home/lih30/allinone/bin/activate
echo "torch before:" $(python -c "import torch; print(torch.__version__)")
pip install "torchio==0.19.6"
echo "torch after:" $(python -c "import torch; print(torch.__version__)")
python -c "import torchio; print(\"torchio\", torchio.__version__, \"OK\")"
echo DONE
