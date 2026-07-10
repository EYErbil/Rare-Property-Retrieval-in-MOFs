#!/bin/bash
#SBATCH --job-name=screening
#SBATCH --output=logs/screening_%j.out
#SBATCH --error=logs/screening_%j.err
#SBATCH --partition=mid
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# OPTIONAL: Two-Signal Candidate Screening (NN + kNN)
# =============================================================================
# Selects MOF candidates using a two-signal approach: union of NN top-K and
# kNN structural-similarity top-K for robust candidate selection.
#
# USAGE:
#   sbatch scripts/optional/run_screening.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

python src/pormake_screen.py \
    --experiments_dir "$EXP_BASE" \
    --embeddings_path "$EMB_FILE" \
    --splits_dir "$SPLITS_DIR" \
    --output_dir "$DATA_DIR/screening_results"

echo "  Output: $DATA_DIR/screening_results/"
