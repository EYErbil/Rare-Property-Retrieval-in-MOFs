#!/bin/bash
#SBATCH --job-name=model_cmp
#SBATCH --output=logs/model_cmp_%j.out
#SBATCH --error=logs/model_cmp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --time=04:00:00
#SBATCH --mem=16G

# =============================================================================
# OPTIONAL: NN vs ML Model Comparison (UMAP Investigation)
# =============================================================================
#
# UMAP-based analysis of where each model's top-K candidates sit in embedding
# space, and whether NN and ML models agree on candidate selection.
#
# PREREQUISITES:
#   - Step 06 completed (embeddings + predictions for unlabeled set)
#
# USAGE:
#   sbatch scripts/optional/run_model_comparison.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

section "MODEL COMPARISON (NN vs ML)"

python discovery/plot_model_comparison.py \
    --embeddings_npz "$DISCOVERY_DATA/embedding_analysis/unlabeled_embeddings.npz" \
    --top_k 25

section "COMPLETE"
echo "  Output: model_comparison/"
