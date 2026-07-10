#!/bin/bash
#SBATCH --job-name=umap_nom
#SBATCH --output=logs/umap_nom_%j.out
#SBATCH --error=logs/umap_nom_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=2:00:00

set -euo pipefail

module purge
module load cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5
source /path/to/your/venv/bin/activate   # edit to your Python environment


python figure_nominated_bandgap_umap.py \
    --pretrained_npz /path/to/project/embeddings/pmt_embeddings_qmof_all.npz \
    --finetuned_npz /path/to/project/posttrain_umap_figures_exp364/posttrain_embeddings.npz \
    --finetuned_name exp364 \
    --bandgap_csv /path/to/project/result-analysis/bandgap_results.csv \
    --labeled_splits_dir /path/to/project/new_splits/strategy_d_farthest_point \
    --qmof_csv /path/to/project/qmof.csv \
    --output_dir ./nominated_umap_figures