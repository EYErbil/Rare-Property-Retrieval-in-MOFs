#!/bin/bash
#SBATCH --job-name=fig_soap_umap
#SBATCH --partition=mid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fig_soap_umap_%j.out
#SBATCH --error=logs/fig_soap_umap_%j.err
set -euo pipefail

# F4: Compute SOAP descriptors from CIF files + generate 4-panel UMAP
#     SOAP is a purely geometry-based structural fingerprint — no NN involved.
#   (a) Labeled vs unlabeled
#   (b) DFT bandgap (discrete bins)
#   (c) Train / val / test splits
#   (d) Ensemble nominations (if provided)
#
# CPU only. High memory for SOAP computation (~20K structures).
# Independent of F1-F3 — only needs CIF files + split JSONs.
# SOAP descriptors are cached after the first run for fast re-runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$FIGURES_OUTPUT/soap_umap"

# --- Configuration -----------------------------------------------------------
UNLABELED_JSON="$DISCOVERY_DATA/test_bandgaps_regression.json"
OUTPUT="$FIGURES_OUTPUT/soap_umap"
# Optional: path to top-25 ensemble nominations file (one CIF ID per line)
NOMINATIONS=""  # e.g., "$DISCOVERY_DATA/nomination-SOAP/FINAL_TOP25_diverse.txt"

section "F4: SOAP descriptors from CIF → UMAP"

EXTRA_ARGS=()
if [ -n "$NOMINATIONS" ] && [ -f "$NOMINATIONS" ]; then
    EXTRA_ARGS+=(--nominations "$NOMINATIONS")
fi

python figures/soap_descriptors_umap.py \
    --cif_dir "$CIF_DIR" \
    --labeled_splits_dir "$SPLITS_DIR" \
    --unlabeled_json "$UNLABELED_JSON" \
    --output_dir "$OUTPUT" \
    --save_umap_cache \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. SOAP UMAP panels saved to $OUTPUT/"
