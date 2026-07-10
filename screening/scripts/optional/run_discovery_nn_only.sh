#!/bin/bash
#SBATCH --job-name=discovery_nn
#SBATCH --output=logs/discovery_nn_%j.out
#SBATCH --error=logs/discovery_nn_%j.err
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# =============================================================================
# OPTIONAL: NN-Only Inference on Unlabeled Data
# =============================================================================
#
# Runs ONLY the neural network models on a new unlabeled MOF dataset.
# Requires a GPU. Use this when you want to run NN inference independently.
#
# PREREQUISITES:
#   - Step 02 completed (NN experiments have checkpoints)
#   - New MOF structures in data/unlabeled/ (.grid, .griddata16, .graphdata)
#   - data/unlabeled/test_bandgaps_regression.json
#
# USAGE:
#   sbatch scripts/optional/run_discovery_nn_only.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$DISCOVERY_DATA/inference_results"

NN_EXPERIMENTS="exp364_fulltune exp370_seed2 exp371_seed3"
TOP_K=25

section "NN INFERENCE ON UNLABELED DATA"
echo "  Experiments: $NN_EXPERIMENTS"
echo "  Data dir:    $DISCOVERY_DATA"
echo "  Top-K:       $TOP_K"

for exp_name in $NN_EXPERIMENTS; do
    exp_dir="$EXP_BASE/$exp_name"

    if [ ! -d "$exp_dir" ]; then
        echo "  WARNING: $exp_dir does not exist. Skipping."
        continue
    fi

    ckpt=$(ls "$exp_dir"/best_es-*.ckpt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        echo "  WARNING: No checkpoint in $exp_dir. Skipping."
        continue
    fi

    section "INFERRING: $exp_name"
    echo "  Checkpoint: $(basename $ckpt)"

    cd "$exp_dir"
    python "$BASE_DIR/discovery/run_inference_from_cwd.py" \
        --data_dir "$DISCOVERY_DATA" \
        --top_k "$TOP_K"
    cd "$BASE_DIR"

    if [ -f "$exp_dir/inference_predictions.csv" ]; then
        echo "  OK: $exp_dir/inference_predictions.csv"
    else
        echo "  WARNING: inference_predictions.csv not produced for $exp_name"
    fi
done

section "NN INFERENCE COMPLETE"
for exp_name in $NN_EXPERIMENTS; do
    csv="$EXP_BASE/$exp_name/inference_predictions.csv"
    if [ -f "$csv" ]; then
        echo "  [OK]   $exp_name"
    else
        echo "  [MISS] $exp_name"
    fi
done
