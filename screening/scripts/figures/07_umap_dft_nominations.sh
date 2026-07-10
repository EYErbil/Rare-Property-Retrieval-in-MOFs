#!/bin/bash
#SBATCH --job-name=fig_nominated
#SBATCH --partition=mid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig_nominated_%j.out
#SBATCH --error=logs/fig_nominated_%j.err
set -euo pipefail

# F7: Plot the 25 nominated structures on UMAP, colored by their DFT bandgap
#     Uses pretrained embeddings (from F1). Optionally overlays a fine-tuned
#     model for side-by-side comparison (requires F3).
#
# CPU only. Requires F1 output + bandgap_results.csv from DFT calculations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$FIGURES_OUTPUT/nominated_umap"

# --- Configuration -----------------------------------------------------------
PRETRAINED_NPZ="$FIGURES_OUTPUT/pretrained_embeddings/all_embeddings.npz"
BANDGAP_CSV="$DISCOVERY_DATA/nomination-SOAP/bandgap_results.csv"
OUTPUT="$FIGURES_OUTPUT/nominated_umap"
# Optional: fine-tuned embeddings for a second set of panels
FINETUNED_NPZ=""  # e.g., "$FIGURES_OUTPUT/finetuned_umap_exp370_seed2/posttrain_embeddings.npz"
FINETUNED_NAME="exp370"

section "F7: Nominated structures colored by DFT bandgap"

EXTRA_ARGS=()
if [ -n "$FINETUNED_NPZ" ] && [ -f "$FINETUNED_NPZ" ]; then
    EXTRA_ARGS+=(--finetuned_npz "$FINETUNED_NPZ" --finetuned_name "$FINETUNED_NAME")
fi

python figures/umap_dft_nominations.py \
    --pretrained_npz "$PRETRAINED_NPZ" \
    --bandgap_csv "$BANDGAP_CSV" \
    --labeled_splits_dir "$SPLITS_DIR" \
    --output_dir "$OUTPUT" \
    --save_umap_cache \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Nominated structure panels saved to $OUTPUT/"
