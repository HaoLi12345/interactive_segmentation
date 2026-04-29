#!/bin/bash
#SBATCH --job-name=iseg_prism_x86
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%j.out

set -e
source /home/lih30/allinone/bin/activate
cd /home/lih30/interactive_seg
echo "=== current torch ==="
python -c "import torch; print(torch.__version__)"
echo "=== install missing PRISM deps (--no-deps, no torch upgrade) ==="
pip install --no-deps kornia torchio prefetch-generator MedPy
echo "=== verify ==="
python -c "import torch, kornia, torchio, prefetch_generator, medpy; print('torch', torch.__version__); print('kornia', kornia.__version__); print('torchio', torchio.__version__); print('PRISM deps OK')"
echo DONE
