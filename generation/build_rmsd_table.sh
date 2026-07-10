#!/bin/bash
#SBATCH --job-name=rmsd_match
#SBATCH --output=logs/rmsd_match_%j.out
#SBATCH --error=logs/rmsd_match_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=mid
#SBATCH --time=24:00:00

set -euo pipefail

module purge
module load cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5
source /path/to/your/venv/bin/activate   # edit to your Python environment



python ./bulk_pormake_generation/rmsd_calculated_node.py \
    --save data/rmsd_qmof.pickle \
    --bb-dir qmof_bb_dir \
    --topo-dir qmof_topo_dir