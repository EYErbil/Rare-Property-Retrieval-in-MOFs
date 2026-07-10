#!/bin/bash
#SBATCH --job-name=train_ml
#SBATCH --output=logs/03_train_ml_%j.out
#SBATCH --error=logs/03_train_ml_%j.err
#SBATCH --partition=mid
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# STEP 3: Train ML Classifiers + kNN Baselines on Pretrained Embeddings
# =============================================================================
#
# Trains 15+ scikit-learn classifiers directly on the 768-dim pretrained
# MOFTransformer embeddings. No GPU needed — these are feature-based models.
#
# ML METHODS (trained by embedding_classifier.py --enhanced):
#   Base:     Logistic Regression, SVM-RBF, Random Forest, Extra Trees,
#             Gradient Boosting, XGBoost, LDA, Mahalanobis, GMM, Isolation Forest
#   Enhanced: SMOTE variants, Two-stage (kNN pre-filter → ET), Feature-selected
#
# kNN BASELINES (trained by knn_baseline.py):
#   - k-NN regression (distance-weighted neighbor bandgap averaging)
#   - Similarity-to-positive (max cosine sim to any train positive)
#   - Hybrid (NN + kNN score combination)
#   - Novelty-aware (flag far-from-training MOFs)
#
# OUTPUT:
#   embedding_classifiers/strategy_d_farthest_point/{method}/
#     ├── model.joblib, scaler.joblib   — Saved model artifacts
#     ├── test_predictions.csv          — Per-MOF scores (used by ensemble)
#     └── final_results.json            — Discovery metrics
#   knn_results/strategy_d_farthest_point/
#     ├── test_predictions.csv          — kNN predictions
#     └── knn_hybrid_results.json       — Hybrid metrics
#
# PREREQUISITES:
#   - Step 01 completed (embeddings_pretrained.npz + splits exist)
#
# USAGE:
#   sbatch scripts/03_train_ml.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$SKLEARN_DIR" "$KNN_DIR"

# ---- Phase A: Sklearn classifiers ------------------------------------------
section "PHASE A: SKLEARN CLASSIFIERS (base + enhanced)"

python src/embedding_classifier.py \
    --embeddings_path "$EMB_FILE" \
    --output_dir "$SKLEARN_DIR" \
    --labels_dir "$SPLITS_DIR" \
    --threshold 1.0 \
    --enhanced

echo "  Output: $SKLEARN_DIR"
SKLEARN_COUNT=$(find "$SKLEARN_DIR" -name "test_predictions.csv" | wc -l)
echo "  Models with predictions: $SKLEARN_COUNT"

# ---- Phase B: kNN baselines ------------------------------------------------
section "PHASE B: kNN BASELINES"

python src/knn_baseline.py \
    --embeddings_path "$EMB_FILE" \
    --output_dir "$KNN_DIR" \
    --labels_dir "$SPLITS_DIR" \
    --K 10 --threshold 1.0

echo "  Output: $KNN_DIR"

section "STEP 3 COMPLETE — ML + kNN TRAINING"
