#!/bin/bash
#SBATCH --job-name=fig_soap_val
#SBATCH --partition=mid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fig_soap_val_%j.out
#SBATCH --error=logs/fig_soap_val_%j.err
set -euo pipefail

# F5: SOAP structural validation — NN-independent confirmation of data split
#   A. Split coverage (SOAP vs NN similarity to nearest train positive)
#   B. Structure-bandgap correlation (does geometry predict bandgap?)
#   C. Applicability domain for ensemble predictions
#   D. Mantel test (SOAP vs NN distance matrix agreement)
#
# CPU only. Requires F1 output (all_embeddings.npz) + CIF files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$FIGURES_OUTPUT/soap_validation"

# --- Configuration -----------------------------------------------------------
EMBEDDINGS="$FIGURES_OUTPUT/pretrained_embeddings/all_embeddings.npz"
OUTPUT="$FIGURES_OUTPUT/soap_validation"
# Optional: top-25 nominations for applicability domain analysis (C)
NOMINATIONS=""  # e.g., "$DISCOVERY_DATA/nomination-SOAP/FINAL_TOP25_diverse.txt"

section "F5: SOAP structural validation"

EXTRA_ARGS=()
if [ -n "$NOMINATIONS" ] && [ -f "$NOMINATIONS" ]; then
    EXTRA_ARGS+=(--nominations "$NOMINATIONS")
fi

python figures/soap_validation.py \
    --cif_dir "$CIF_DIR" \
    --merged_embeddings "$EMBEDDINGS" \
    --labeled_splits_dir "$SPLITS_DIR" \
    --output_dir "$OUTPUT" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. SOAP validation results saved to $OUTPUT/"
