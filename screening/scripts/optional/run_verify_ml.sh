#!/bin/bash
#SBATCH --job-name=verify_ml
#SBATCH --output=logs/verify_ml_%j.out
#SBATCH --error=logs/verify_ml_%j.err
#SBATCH --partition=mid
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# OPTIONAL: Verify ML Model Reproducibility
# =============================================================================
# Loads all saved sklearn models, re-predicts on test set, and generates a
# per-positive heatmap to verify predictions match saved results exactly.
#
# USAGE:
#   sbatch scripts/optional/run_verify_ml.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

python src/verify_ml_heatmap.py \
    --base_dir "$BASE_DIR" \
    --embeddings_path "$EMB_FILE" \
    --labels_dir "$SPLITS_DIR" \
    --clf_dir "$SKLEARN_DIR" \
    --output_dir "$DATA_DIR/verification_output" \
    --threshold 1.0

echo "  Output: $DATA_DIR/verification_output/"
