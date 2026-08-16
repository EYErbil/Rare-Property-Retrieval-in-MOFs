#!/bin/bash
#SBATCH --job-name=dft_nominate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/07_nomination_%j.out
#SBATCH --error=logs/07_nomination_%j.err
#SBATCH --mail-type=ALL

# Paper candidate-selection entrypoint for the unlabeled QMOF pool.
# Predictive scores come from one fine-tuned PMTransformer regressor and one
# SMOTE--ExtraTrees classifier on frozen PMTransformer embeddings. Candidate
# diversity is evaluated only in SOAP space: this training-free geometric
# coordinate avoids selecting near-duplicates using the same representation
# that supplies the predictive scores. RRF and disagreement remain priority
# signals; neither is treated as a structural-diversity coordinate.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

NN_EXP="${NN_EXP:-exp364_fulltune}"
ML_METHOD="${ML_METHOD:-smote_extra_trees}"
NN_CSV="${NN_CSV:-$EXP_BASE/$NN_EXP/inference_predictions.csv}"
ML_CSV="${ML_CSV:-$DISCOVERY_DATA/ml_predictions/$ML_METHOD/test_predictions.csv}"
SOAP_EMBEDDINGS="${SOAP_EMBEDDINGS:-$BASE_DIR/../generation/soap_analysis/soap_descriptors_sparse.npz}"
SOAP_KEY="${SOAP_KEY:-soap_descriptors}"
OUTPUT_DIR="${OUTPUT_DIR:-$DISCOVERY_DATA/nomination-SOAP}"

for required in "$NN_CSV" "$ML_CSV" "$SOAP_EMBEDDINGS"; do
  if [ ! -f "$required" ]; then
    echo "ERROR: required input not found: $required" >&2
    exit 2
  fi
done

# Exact paper settings. Hidden implementation constants are also exposed here:
# PCA=50, KMeans n_init=10, exploration score weights 0.60/0.40, and
# exploration MMR lambda=0.40.
python discovery/nominate_diverse_dft.py \
  --embeddings_path "$SOAP_EMBEDDINGS" \
  --embedding_key "$SOAP_KEY" \
  --embedding_label SOAP \
  --prediction_csvs \
    "$NN_EXP=$NN_CSV" \
    "$ML_METHOD=$ML_CSV" \
  --nn_models "$NN_EXP" \
  --ml_models "$ML_METHOD" \
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

section "STEP 7 COMPLETE"
echo "SOAP-only nomination: $OUTPUT_DIR/FINAL_TOP25_diverse.txt"
