#!/bin/bash
#SBATCH --job-name=fig_ens_umap
#SBATCH --partition=mid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fig_ens_umap_%j.out
#SBATCH --error=logs/fig_ens_umap_%j.err
set -euo pipefail

# F6: Overlay the top-25 ensemble nominations on per-experiment fine-tuned UMAPs
#     Shows where the nominated structures sit in each model's learned space.
#
# CPU only. Requires F3 outputs (posttrain_embeddings.npz per experiment).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$FIGURES_OUTPUT/ensemble_umap"

# --- Configuration -----------------------------------------------------------
NOMINATIONS="$DISCOVERY_DATA/nomination-SOAP/FINAL_TOP25_diverse.txt"
OUTPUT="$FIGURES_OUTPUT/ensemble_umap"

section "F6: Ensemble nominations on fine-tuned UMAPs"

python figures/umap_ensemble_nominations.py \
    --npz_exp364 "$FIGURES_OUTPUT/finetuned_umap_exp364_fulltune/posttrain_embeddings.npz" \
    --npz_exp370 "$FIGURES_OUTPUT/finetuned_umap_exp370_seed2/posttrain_embeddings.npz" \
    --npz_exp371 "$FIGURES_OUTPUT/finetuned_umap_exp371_seed3/posttrain_embeddings.npz" \
    --labeled_splits_dir "$SPLITS_DIR" \
    --nominations "$NOMINATIONS" \
    --output_dir "$OUTPUT" \
    --load_umap_cache \
    --save_umap_cache

echo ""
echo "Done. Ensemble UMAP panels saved to $OUTPUT/"
