#!/bin/bash
#SBATCH --job-name=exp371
#SBATCH --output=logs/exp371_%j.out
#SBATCH --error=logs/exp371_%j.err
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --time=69:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "=== [JOB STARTED] ==="
date
echo "Running on node: $(hostname)"

# Load project configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/config.sh"
load_modules

echo "Exp 371: Split D fulltune seed=456 (ensemble diversity)"

EXP_DIR="$REPO_ROOT/experiments/exp371_seed3"
cd "$EXP_DIR"
mkdir -p logs
python run.py --data_dir "$SPLITS_DIR"

echo "=== [JOB COMPLETED] ==="
date
