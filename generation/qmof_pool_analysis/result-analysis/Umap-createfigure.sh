#!/bin/bash
#SBATCH --job-name=UMAP
#SBATCH --output=logs/UMAP_%j.out
#SBATCH --error=logs/UMAP_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short
#SBATCH --time=2:00:00


set -euo pipefail

module purge
module load cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5
source /path/to/your/venv/bin/activate   # edit to your Python environment

python figure2_chemical_space_umap.py \
  --merged_embeddings /path/to/project/embeddings/pmt_embeddings_qmof_all.npz \
  --labeled_splits_dir /path/to/project/new_splits/strategy_d_farthest_point \
  --qmof_csv /path/to/project/qmof.csv \
  --output_dir ./paper_figures \
  --save_cache