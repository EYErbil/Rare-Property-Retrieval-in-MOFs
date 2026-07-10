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

# =============================================================================
# STEP 7: Diversity-Aware DFT Candidate Nomination
# =============================================================================
#
# Selects the final top-25 structures for DFT bandgap calculation using a
# diversity-aware pipeline:
#   1. RRF fusion of 1 NN + 1 ML model predictions
#   2. Cluster the top-500 shortlist in embedding space
#   3. Four diversity-aware strategies (cluster quota, MMR, uncertainty-
#      weighted quota, long-tail exploration) produce nominees
#   4. Combined best-of-all final 25
#
# Two runs are performed: one using PMTransformer embeddings for diversity,
# one using SOAP descriptors. SOAP-based diversity produces better structural
# spread because it measures purely geometric/chemical similarity independent
# of the learned representations used for scoring.
#
# PREREQUISITES:
#   - Steps 02-03 completed (trained models exist)
#   - Step 06 completed (embeddings + predictions for unlabeled set)
#   - SOAP descriptors computed (soap_descriptors.npz)
#
# CONFIGURATION:
#   Edit the variables below to match your model paths.
#
# USAGE:
#   sbatch scripts/07_nominate_candidates.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

# --- Configuration ----------------------------------------------------------
# Paths to model predictions (1 NN + 1 ML recommended)
NN_EXP="exp364_fulltune"
ML_METHOD="extra_trees"

NN_CSV="$EXP_BASE/$NN_EXP/inference_predictions.csv"
ML_CSV="$SKLEARN_DIR/$ML_METHOD/test_predictions.csv"

# Embeddings for diversity computation
PMT_EMBEDDINGS="$DISCOVERY_DATA/embedding_analysis/unlabeled_embeddings.npz"
SOAP_EMBEDDINGS=""  # Set to path of soap_descriptors.npz if available

# Optional: path to a previous nomination for comparison
OLD_NOMINEES=""  # e.g., "$DISCOVERY_DATA/previous_nomination/FINAL_TOP25_diverse.txt"

# Nomination parameters
POOL_SIZE=500
N_CLUSTERS=20
MAX_PER_CLUSTER=1
MMR_LAMBDAS="0.2 0.3 0.4"
BUDGET=25
EXPLORATION_BUDGET=5
EXPLORATION_POOL_HI=2000
RRF_K=60
SEED=42

# Shared arguments
COMMON_ARGS=(
  --prediction_csvs
      "${NN_EXP}=${NN_CSV}"
      "${ML_METHOD}=${ML_CSV}"
  --nn_models "$NN_EXP"
  --ml_models "$ML_METHOD"
  --pool_size "$POOL_SIZE"
  --n_clusters "$N_CLUSTERS"
  --max_per_cluster "$MAX_PER_CLUSTER"
  --mmr_lambdas $MMR_LAMBDAS
  --budget "$BUDGET"
  --exploration_budget "$EXPLORATION_BUDGET"
  --exploration_pool_hi "$EXPLORATION_POOL_HI"
  --rrf_k "$RRF_K"
  --seed "$SEED"
)

# Add old nominees if specified
if [ -n "$OLD_NOMINEES" ] && [ -f "$OLD_NOMINEES" ]; then
  COMMON_ARGS+=(--old_nominees "$OLD_NOMINEES")
fi

# =============================================================================
# RUN 1: PMTransformer diversity
# =============================================================================
section "RUN 1: PMTransformer diversity"

SOAP_ARG=""
if [ -n "$SOAP_EMBEDDINGS" ] && [ -f "$SOAP_EMBEDDINGS" ]; then
  SOAP_ARG="--soap_embeddings_path $SOAP_EMBEDDINGS"
fi

python discovery/nominate_diverse_dft.py \
  --embeddings_path "$PMT_EMBEDDINGS" \
  --embedding_key embeddings \
  --embedding_label PMTransformer \
  $SOAP_ARG \
  --output_dir "$DISCOVERY_DATA/nomination-PMT" \
  "${COMMON_ARGS[@]}"

# =============================================================================
# RUN 2: SOAP diversity (if SOAP embeddings available)
# =============================================================================
if [ -n "$SOAP_EMBEDDINGS" ] && [ -f "$SOAP_EMBEDDINGS" ]; then
  section "RUN 2: SOAP diversity"

  python discovery/nominate_diverse_dft.py \
    --embeddings_path "$SOAP_EMBEDDINGS" \
    --embedding_key soap_descriptors \
    --embedding_label SOAP \
    --output_dir "$DISCOVERY_DATA/nomination-SOAP" \
    "${COMMON_ARGS[@]}"
else
  echo ""
  echo "  Skipping SOAP run (no SOAP embeddings found at: $SOAP_EMBEDDINGS)"
  echo "  To enable: set SOAP_EMBEDDINGS at the top of this script."
fi

# =============================================================================
section "STEP 7 COMPLETE"
echo ""
echo "  PMTransformer nomination: $DISCOVERY_DATA/nomination-PMT/"
if [ -n "$SOAP_EMBEDDINGS" ] && [ -f "$SOAP_EMBEDDINGS" ]; then
  echo "  SOAP nomination:          $DISCOVERY_DATA/nomination-SOAP/"
fi
echo ""
echo "  Compare the two to see how SOAP diversity improves structural spread."
