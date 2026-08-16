#!/bin/bash
# Paper candidate-selection entrypoint for the generated framework pool.
# RRF/disagreement supply priority; SOAP alone supplies the PCA, clustering,
# pairwise-distance, and MMR geometry. Protocol exclusions (including the two
# lanthanide candidates in the recorded campaign) are applied only after the
# 25-member nomination has been written.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$GEN_ROOT"

NN_CSV="${NN_CSV:-$GEN_ROOT/re_infer/nn/exp364/inference_predictions.csv}"
ML_CSV="${ML_CSV:-$GEN_ROOT/re_infer/ml/smote_extra_trees/test_predictions.csv}"
SOAP_EMBEDDINGS="${SOAP_EMBEDDINGS:-$GEN_ROOT/soap_analysis/generated_vs_qmof/generated_soap_descriptors.npz}"
SOAP_KEY="${SOAP_KEY:-soap_descriptors}"
OUTPUT_DIR="${OUTPUT_DIR:-$GEN_ROOT/paper_results/nomination-SOAP}"

for required in "$NN_CSV" "$ML_CSV" "$SOAP_EMBEDDINGS"; do
  if [ ! -f "$required" ]; then
    echo "ERROR: required input not found: $required" >&2
    exit 2
  fi
done

python nominate_diverse_dft.py \
  --embeddings_path "$SOAP_EMBEDDINGS" \
  --embedding_key "$SOAP_KEY" \
  --embedding_label SOAP \
  --prediction_csvs \
    "exp364=$NN_CSV" \
    "smote_extra_trees=$ML_CSV" \
  --nn_models exp364 \
  --ml_models smote_extra_trees \
  --output_dir "$OUTPUT_DIR" \
  --pool_size 500 \
  --pca_components 50 \
  --n_clusters 20 \
  --kmeans_n_init 10 \
  --max_per_cluster 1 \
  --mmr_lambdas 0.2 0.3 0.4 \
  --alpha 0.50 \
  --beta 0.30 \
  --gamma 0.20 \
  --budget 25 \
  --exploration_budget 5 \
  --exploration_pool_lo 500 \
  --exploration_pool_hi 2000 \
  --exploration_disagreement_weight 0.60 \
  --exploration_rank_std_weight 0.40 \
  --exploration_mmr_lambda 0.40 \
  --rrf_k 60 \
  --seed 42

echo "SOAP-only nomination: $OUTPUT_DIR/FINAL_TOP25_diverse.txt"
