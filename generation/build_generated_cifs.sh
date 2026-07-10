#!/bin/bash
#SBATCH --job-name=build_cifs
#SBATCH --output=logs/build_cifs_%j.out
#SBATCH --error=logs/build_cifs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=mid
#SBATCH --time=24:00:00

set -euo pipefail

module purge
module load cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5
source /path/to/your/venv/bin/activate   # edit to your Python environment



python scripts/build_materials_batched.py \
    --candidates data/candidates_qmof_13k_200atom.txt \
    --bb-dir     qmof_bb_dir \
    --topo-dir   qmof_topo_dir \
    --save-dir   generated_cifs/small_30A_200atom \
    --large-dir  generated_cifs/large_30A_200atom \
    --cutoff     30.0 \
    --chunk-size 200 \
    --continue-on-error