#!/bin/bash
#SBATCH --job-name=soap_umap
#SBATCH --output=logs/soap_umap_%j.out
#SBATCH --error=logs/soap_umap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --time=24:00:00

set -euo pipefail

module purge
module load cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5
source /path/to/your/venv/bin/activate   # edit to your Python environment


python figure_soap_umap.py \
    --cif_dir /path/to/project/qmof_test_cifs \
    --labeled_splits_dir /path/to/project/new_splits/strategy_d_farthest_point  \
    --unlabeled_json /path/to/project/unlabeled/test_bandgaps_regression.json \
    --soap_cache /path/to/project/soap_analysis/soap_descriptors.npz \
    --nominated_top_predictions /path/to/project/result-analysis/nominated.txt \
    --bandgap_csv /path/to/project/result-analysis/bandgap_results.csv \
    --output_dir ./soap_umap_figures \
    --save_umap_cache