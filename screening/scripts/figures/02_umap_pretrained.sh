#!/bin/bash
#SBATCH --job-name=fig_pretrained
#SBATCH --partition=mid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig_pretrained_%j.out
#SBATCH --error=logs/fig_pretrained_%j.err
set -euo pipefail

# F2: Generate 4-panel UMAP from PRETRAINED PMTransformer embeddings
#   (a) Labeled vs unlabeled
#   (b) DFT bandgap (discrete bins)
#   (c) Primary metal center (from qmof.csv)
#   (d) Labeled-only zoomed view: train / val / test
#
# CPU only. Requires F1 output (all_embeddings.npz).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$FIGURES_OUTPUT/pretrained_umap"

# --- Configuration -----------------------------------------------------------
EMBEDDINGS="$FIGURES_OUTPUT/pretrained_embeddings/all_embeddings.npz"
OUTPUT="$FIGURES_OUTPUT/pretrained_umap"

section "F2: UMAP of pretrained PMTransformer embeddings"

python figures/umap_pretrained.py \
    --merged_embeddings "$EMBEDDINGS" \
    --labeled_splits_dir "$SPLITS_DIR" \
    --qmof_csv "$QMOF_CSV" \
    --output_dir "$OUTPUT" \
    --save_cache

echo ""
echo "Done. 4 panels saved to $OUTPUT/"
