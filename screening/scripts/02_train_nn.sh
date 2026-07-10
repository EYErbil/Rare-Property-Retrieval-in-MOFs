#!/bin/bash
#SBATCH --job-name=train_nn
#SBATCH --output=logs/02_train_nn_%j.out
#SBATCH --error=logs/02_train_nn_%j.err
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --time=69:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# =============================================================================
# STEP 2: Train Neural Network Regressors (MOFTransformer Fine-Tuning)
# =============================================================================
#
# Trains MOFTransformer-based bandgap regressors. Each experiment uses the same
# architecture but different random seeds for ensemble diversity:
#
#   exp364 — Full finetune, seed=42  (primary model)
#   exp370 — Full finetune, seed=123 (ensemble variant)
#   exp371 — Full finetune, seed=456 (ensemble variant)
#
# All experiments use:
#   - Huber loss (robust to outlier bandgaps)
#   - Early stopping on val/spearman_rho (ranking quality, not loss)
#   - Mean pooling over atom tokens (not CLS)
#   - lr=1e-4 with 10x differential LR for regression head
#
# OUTPUT (per experiment):
#   - best_es-*.ckpt        — Best checkpoint (Spearman ρ)
#   - test_predictions.csv  — Per-MOF predictions (used by ensemble)
#   - final_results.json    — All test metrics
#   - training_dashboard.png, discovery_curve.png, etc.
#
# PREREQUISITES:
#   - Step 01 completed (embeddings + splits exist)
#   - GPU node available
#
# USAGE:
#   sbatch scripts/02_train_nn.sh                      # Run all 3 sequentially
#   sbatch scripts/02_train_nn.sh --exp exp364_fulltune # Run one experiment
#
# NOTE: For parallel training, submit each experiment separately:
#   cd experiments/exp364_fulltune && sbatch run.sh
#   cd experiments/exp370_seed2   && sbatch run.sh
#   cd experiments/exp371_seed3   && sbatch run.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs

# Parse optional --exp argument
TARGET_EXP="${1:-all}"

train_experiment() {
    local exp_name="$1"
    local exp_dir="$EXP_BASE/$exp_name"

    section "TRAINING: $exp_name"

    if [ ! -f "$exp_dir/run.py" ]; then
        echo "  ERROR: $exp_dir/run.py not found. Skipping."
        return 1
    fi

    cd "$exp_dir"
    mkdir -p logs

    python run.py --data_dir "$SPLITS_DIR"

    echo "  Checkpoint: $(ls best_es-*.ckpt 2>/dev/null || echo 'none')"
    echo "  Predictions: $([ -f test_predictions.csv ] && echo 'OK' || echo 'missing')"

    cd "$BASE_DIR"
}

if [ "$TARGET_EXP" = "all" ] || [ "$TARGET_EXP" = "--exp" ]; then
    if [ "$TARGET_EXP" = "--exp" ]; then
        TARGET_EXP="${2:-all}"
    fi
fi

if [ "$TARGET_EXP" = "all" ]; then
    for exp in exp364_fulltune exp370_seed2 exp371_seed3; do
        train_experiment "$exp"
    done
else
    train_experiment "$TARGET_EXP"
fi

section "STEP 2 COMPLETE — NN TRAINING"
echo "  Experiments trained. Check experiments/*/final_results.json for metrics."
