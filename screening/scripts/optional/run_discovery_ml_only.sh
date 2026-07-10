#!/bin/bash
#SBATCH --job-name=discovery_ml
#SBATCH --output=logs/discovery_ml_%j.out
#SBATCH --error=logs/discovery_ml_%j.err
#SBATCH --partition=mid
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# OPTIONAL: ML-Only Inference on Unlabeled Data
# =============================================================================
#
# Runs ONLY the ML classifiers (sklearn models) on a new unlabeled dataset.
# No GPU needed. Equivalent to Steps 6a + 6b from 06_run_discovery.sh.
#
# PREREQUISITES:
#   - Step 03 completed (sklearn models saved)
#   - New MOF structures in data/unlabeled/ (.grid, .griddata16, .graphdata)
#   - data/unlabeled/test_bandgaps_regression.json
#
# USAGE:
#   sbatch scripts/optional/run_discovery_ml_only.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$DISCOVERY_DATA/embedding_analysis" "$DISCOVERY_DATA/ml_predictions"

section "EXTRACT UNLABELED EMBEDDINGS"

python data_preparation/extract_unlabeled_embeddings.py \
    --data_dir "$DISCOVERY_DATA" \
    --output_dir "$DISCOVERY_DATA/embedding_analysis"

DISCOVERY_NPZ="$DISCOVERY_DATA/embedding_analysis/unlabeled_embeddings.npz"
echo "  Output: $DISCOVERY_NPZ"

section "ML INFERENCE ON UNLABELED DATA"

python src/predict_with_embedding_classifier.py \
    --embeddings_path "$DISCOVERY_NPZ" \
    --models_dir "$SKLEARN_DIR" \
    --output_dir "$DISCOVERY_DATA/ml_predictions"

ML_PRED_COUNT=$(find "$DISCOVERY_DATA/ml_predictions" -name "test_predictions.csv" 2>/dev/null | wc -l)
echo "  ML methods with predictions: $ML_PRED_COUNT"

NN_CSV="$DISCOVERY_DATA/inference_results/inference_predictions.csv"
if [ -f "$NN_CSV" ]; then
    section "RUNNING ML + NN DISCOVERY (NN predictions found)"
    python discovery/discovery_pipeline.py \
        --discovery_dir "$DISCOVERY_DATA" \
        --embeddings_path "$DISCOVERY_NPZ" \
        --ml_models_dir "$SKLEARN_DIR" \
        --output_dir "$DISCOVERY_DATA/discovery_output" \
        --rrf_k 60 \
        --nn_predictions "$NN_CSV"
else
    section "RUNNING ML-ONLY DISCOVERY (no NN predictions yet)"
    python discovery/discovery_pipeline.py \
        --discovery_dir "$DISCOVERY_DATA" \
        --embeddings_path "$DISCOVERY_NPZ" \
        --ml_models_dir "$SKLEARN_DIR" \
        --output_dir "$DISCOVERY_DATA/discovery_output" \
        --rrf_k 60
fi

section "ML-ONLY INFERENCE COMPLETE"
echo "  Predictions: $DISCOVERY_DATA/ml_predictions/"
echo "  Discovery output: $DISCOVERY_DATA/discovery_output/"
