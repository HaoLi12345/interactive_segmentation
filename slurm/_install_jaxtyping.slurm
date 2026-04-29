#!/bin/bash
#SBATCH --job-name=iseg_jt
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=outputs/slurm/%j.out
set -e
source /home/lih30/allinone/bin/activate
pip install --no-deps jaxtyping
python -c "import torchio; print(\"torchio\", torchio.__version__, \"OK\")"
echo DONE
