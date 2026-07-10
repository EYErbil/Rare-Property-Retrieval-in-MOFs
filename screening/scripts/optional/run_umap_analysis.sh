#!/bin/bash
#SBATCH --job-name=umap
#SBATCH --output=logs/umap_%j.out
#SBATCH --error=logs/umap_%j.err
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# OPTIONAL: UMAP Embedding Analysis
# =============================================================================
# Generates UMAP visualizations of pretrained and post-training embeddings.
# Useful for diagnosing split quality and understanding model behavior.
#
# USAGE:
#   sbatch scripts/optional/run_umap_analysis.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$DATA_DIR/umap_analysis"

section "UMAP: ORIGINAL vs SPLIT D DIAGNOSIS"

python src/umap_analysis_original_split.py \
    --embeddings_path "$EMB_FILE" \
    --original_splits_dir "$DATA_DIR/splits/original" \
    --split_d_dir "$SPLITS_DIR" \
    --output_dir "$DATA_DIR/umap_analysis/original_split"

section "UMAP: SPLIT D ANALYSIS"

python src/umap_analysis_split_d.py \
    --embeddings_path "$EMB_FILE" \
    --splits_dir "$SPLITS_DIR" \
    --output_dir "$DATA_DIR/umap_analysis/split_d"

section "UMAP: POST-TRAINING EMBEDDINGS (exp364)"

python src/extract_posttrain_embeddings.py \
    --checkpoint "$EXP_BASE/exp364_fulltune/best_es-*.ckpt" \
    --data_dir "$SPLITS_DIR" \
    --output_dir "$DATA_DIR/umap_analysis/posttrain_exp364"

python src/umap_posttrain.py \
    --embeddings_path "$DATA_DIR/umap_analysis/posttrain_exp364/embeddings_posttrain.npz" \
    --output_dir "$DATA_DIR/umap_analysis/posttrain_exp364"

section "UMAP ANALYSIS COMPLETE"
echo "  Output: $DATA_DIR/umap_analysis/"
