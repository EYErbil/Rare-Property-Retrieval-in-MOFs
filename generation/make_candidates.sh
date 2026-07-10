#!/bin/bash
#SBATCH --job-name=candidates
#SBATCH --output=logs/candidates_%j.out
#SBATCH --error=logs/candidates_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --time=24:00:00

set -euo pipefail

module purge
module load cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5
source /path/to/your/venv/bin/activate   # edit to your Python environment



python ./bulk_pormake_generation/make_candidates.py \
    -n 20000 \
    --max-n-atoms 200 \
    --pre-defined-list data/rmsd_qmof.pickle \
    --save data/candidates_qmof_13k_200atom.txt \
    --bb-dir qmof_bb_dir \
    --topo-dir qmof_topo_dir