#!/bin/bash
#SBATCH --job-name=fig_finetuned
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fig_finetuned_%j.out
#SBATCH --error=logs/fig_finetuned_%j.err
set -euo pipefail

# F3: Run FINE-TUNED PMTransformer forward pass on ALL MOFs, then generate
#     the same 4-panel UMAP as F2 — this time in the fine-tuned embedding space.
#     Lets you see how training changed the representation.
#
# GPU required for the forward pass (--extract). Plotting (--plot) is CPU.
# Runs for each experiment in EXPERIMENTS list.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

# --- Configuration -----------------------------------------------------------
DATA_DIR_MOFS="$DATA_DIR/raw/test"
UNLABELED_JSON="$DISCOVERY_DATA/test_bandgaps_regression.json"
EXPERIMENTS=("exp364_fulltune" "exp370_seed2" "exp371_seed3")

for EXP in "${EXPERIMENTS[@]}"; do
    section "F3: Fine-tuned forward pass + UMAP — $EXP"

    OUT="$FIGURES_OUTPUT/finetuned_umap_${EXP}"
    mkdir -p "$OUT"

    python figures/forward_finetuned_umap.py \
        --extract --plot \
        --experiment "$EXP" \
        --data_dir "$DATA_DIR_MOFS" \
        --labeled_splits_dir "$SPLITS_DIR" \
        --unlabeled_json "$UNLABELED_JSON" \
        --qmof_csv "$QMOF_CSV" \
        --output_dir "$OUT" \
        --save_umap_cache

    echo "  $EXP done -> $OUT/"
done

echo ""
echo "Done. Fine-tuned embeddings + UMAP panels for ${#EXPERIMENTS[@]} experiments."
