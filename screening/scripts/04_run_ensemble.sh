#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --output=logs/04_ensemble_%j.out
#SBATCH --error=logs/04_ensemble_%j.err
#SBATCH --partition=mid
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# =============================================================================
# STEP 4: Exhaustive Ensemble Discovery
# =============================================================================
#
# Combines ALL trained models (NN + sklearn + kNN) and runs exhaustive ablation
# to find the optimal model subset for MOF discovery.
#
# WHAT IT DOES:
#   Phase A: Verify sklearn classifiers exist (error if missing)
#   Phase B: Verify kNN baselines exist (error if missing)
#   Phase C: EXHAUSTIVE ensemble — tests ALL 2/3/4-model combinations
#            Optimizes recall@50 (maximize true positives in top 50 candidates)
#   Phase D: SELECTIVE ensemble — only signal-bearing models (NN + top ML)
#   Phase E: Comparison report across all methods
#
# ENSEMBLE METHODS (per combination):
#   - Reciprocal Rank Fusion (RRF, k=60)
#   - Rank averaging
#   - Top-K voting
#   - Score averaging
#   - Weighted RRF
#
# OUTPUT:
#   ensemble_results/exhaustive/{RRF,rank_avg,...}/
#     ├── ensemble_results.json         — All metrics + per-positive ranks
#     ├── recommended_combinations.txt  — Best 2/3/4-model combos
#     └── top{25,50,100}_for_discovery.txt
#   ensemble_results/selective/          — Same structure, signal-bearing only
#   comparison_report/                   — Cross-method comparison
#
# PREREQUISITES:
#   - Step 02 completed (NN experiments have test_predictions.csv)
#   - Step 03 completed (sklearn + kNN have test_predictions.csv)
#
# USAGE:
#   sbatch scripts/04_run_ensemble.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$ENSEMBLE_DIR/exhaustive" "$ENSEMBLE_DIR/selective"

# =============================================================================
# PREREQUISITES CHECK
# =============================================================================
section "CHECKING PREREQUISITES"

if [ ! -f "$EMB_FILE" ]; then
    echo "ERROR: Embeddings not found at $EMB_FILE"
    echo "  Run 01_extract_embeddings.sh first."
    exit 1
fi
echo "  Embeddings: OK"

NN_COUNT=0
for exp_dir in "$EXP_BASE"/exp*/; do
    if [ -f "${exp_dir}test_predictions.csv" ]; then
        NN_COUNT=$((NN_COUNT + 1))
        echo "  [NN] OK  $(basename "$exp_dir")"
    fi
done
echo "  NN models: $NN_COUNT"

if [ ! -d "$SKLEARN_DIR" ] || [ "$(find "$SKLEARN_DIR" -name 'test_predictions.csv' 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "ERROR: No sklearn classifiers found. Run 03_train_ml.sh first."
    exit 1
fi
echo "  Sklearn: OK"

# =============================================================================
# PHASE C: EXHAUSTIVE ENSEMBLE
# =============================================================================
section "PHASE C: EXHAUSTIVE ENSEMBLE DISCOVERY"

# Collect NN prediction directories
NN_DIRS=""
for exp_dir in "$EXP_BASE"/exp*/; do
    if [ -f "${exp_dir}test_predictions.csv" ]; then
        NN_DIRS="$NN_DIRS $exp_dir"
    fi
done

python src/ensemble_discovery.py \
    --auto_discover \
    --nn_dirs $NN_DIRS \
    --clf_dir "$SKLEARN_DIR" \
    --knn_dir "$KNN_DIR" \
    --output_dir "$ENSEMBLE_DIR/exhaustive" \
    --threshold 1.0 \
    --rrf_k 60 \
    --ablation \
    --ablation_metric recall@50 \
    --recommend_metrics recall@25 recall@50 recall@100 \
    --recommend_max_models 4 \
    --exhaustive_search_limits \
    --search_max_combo_size 4 \
    --subsampled \
    --n_subsample 1500

echo "  Exhaustive ensemble complete: $ENSEMBLE_DIR/exhaustive"

# =============================================================================
# PHASE D: SELECTIVE ENSEMBLE (signal-bearing models only)
# =============================================================================
section "PHASE D: SELECTIVE ENSEMBLE"

PRED_DIRS=""

# NNs
for exp_dir in "$EXP_BASE"/exp*/; do
    if [ -f "$exp_dir/test_predictions.csv" ]; then
        PRED_DIRS="$PRED_DIRS $exp_dir"
        echo "  [NN]  $(basename $exp_dir)"
    fi
done

# Signal-bearing sklearn
for model in extra_trees smote_extra_trees two_stage_knn_et random_forest; do
    model_dir="$SKLEARN_DIR/$model"
    if [ -f "$model_dir/test_predictions.csv" ]; then
        PRED_DIRS="$PRED_DIRS $model_dir"
        echo "  [ML]  $model"
    fi
done

# kNN
if [ -f "$KNN_DIR/test_predictions.csv" ]; then
    PRED_DIRS="$PRED_DIRS $KNN_DIR"
    echo "  [kNN] knn_baseline"
fi

python src/ensemble_discovery.py \
    --prediction_dirs $PRED_DIRS \
    --output_dir "$ENSEMBLE_DIR/selective" \
    --threshold 1.0 \
    --rrf_k 60 \
    --ablation \
    --ablation_metric recall@50 \
    --recommend_metrics recall@25 recall@50 recall@100 \
    --recommend_max_models 4 \
    --max_models 8

echo "  Selective ensemble complete: $ENSEMBLE_DIR/selective"

# =============================================================================
# PHASE E: COMPARISON REPORT
# =============================================================================
section "PHASE E: COMPARISON REPORT"

python src/compare_results.py \
    --experiments_base "$EXP_BASE" \
    --embedding_classifiers "$SKLEARN_DIR" \
    --knn_results "$KNN_DIR" \
    --output_dir "$DATA_DIR/comparison_report" \
    --verbose

section "STEP 4 COMPLETE — ENSEMBLE DISCOVERY"
echo "  Exhaustive results: $ENSEMBLE_DIR/exhaustive/"
echo "  Selective results:  $ENSEMBLE_DIR/selective/"
echo "  Comparison report:  $DATA_DIR/comparison_report/"
