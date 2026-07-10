#!/bin/bash
#SBATCH --job-name=fig_embed
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fig_embed_%j.out
#SBATCH --error=logs/fig_embed_%j.err
set -euo pipefail

# F1: Run pretrained PMTransformer forward pass on ALL MOFs (labeled + unlabeled)
#     in a single run, producing aligned 768-dim embeddings for every structure.
#     GPU required.
#
# Output: $FIGURES_OUTPUT/pretrained_embeddings/all_embeddings.npz
#   Keys: cif_ids, embeddings [N,768], bandgaps, splits, is_labeled
#
# The script auto-detects which MOFs are labeled vs unlabeled from your
# split JSONs — you do NOT need to label anything manually.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$FIGURES_OUTPUT/pretrained_embeddings"

# --- Configuration -----------------------------------------------------------
DATA_DIR_MOFS="$DATA_DIR/raw/test"        # Parent dir containing .graphdata files
UNLABELED_JSON="$DISCOVERY_DATA/test_bandgaps_regression.json"
OUTPUT="$FIGURES_OUTPUT/pretrained_embeddings"

section "F1: Pretrained PMTransformer forward pass → embeddings"

python figures/forward_pretrained_embeddings.py \
    --data_dir "$DATA_DIR_MOFS" \
    --labeled_splits_dir "$SPLITS_DIR" \
    --unlabeled_json "$UNLABELED_JSON" \
    --output_dir "$OUTPUT" \
    --batch_size 1 \
    --num_workers 0

echo ""
echo "Done. Embeddings saved to $OUTPUT/all_embeddings.npz"
echo "Next: run 02_umap_pretrained.sh (CPU) or 03_forward_finetuned_umap.sh (GPU)"
