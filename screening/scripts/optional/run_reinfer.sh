#!/bin/bash
#SBATCH --job-name=reinfer
#SBATCH --output=logs/reinfer_%j.out
#SBATCH --error=logs/reinfer_%j.err
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# =============================================================================
# OPTIONAL: Re-infer NN Predictions After Split Changes
# =============================================================================
# If you modify the train/val/test splits (using tools/), run this to
# re-generate NN test_predictions.csv from existing checkpoints.
#
# USAGE:
#   sbatch scripts/optional/run_reinfer.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

section "RE-INFER NN MODELS"

python src/reinfer_nn.py \
    --experiments_dir "$EXP_BASE" \
    --data_dir "$SPLITS_DIR"

section "RE-INFERENCE COMPLETE"
