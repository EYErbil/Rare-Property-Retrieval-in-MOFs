#!/bin/bash
#SBATCH --job-name=exp364
#SBATCH --output=logs/exp364_%j.out
#SBATCH --error=logs/exp364_%j.err
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --time=69:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load project configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/config.sh"
load_modules

EXP_DIR="$REPO_ROOT/experiments/exp364_fulltune"
cd "$EXP_DIR"
mkdir -p logs
python run.py --data_dir "$SPLITS_DIR"
