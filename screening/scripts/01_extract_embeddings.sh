#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --output=logs/01_embeddings_%j.out
#SBATCH --error=logs/01_embeddings_%j.err
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# =============================================================================
# STEP 1: Extract Pretrained Embeddings & Create Data Splits
# =============================================================================
#
# This step:
#   1a. Extracts 768-dim CLS embeddings from the pretrained MOFTransformer
#       for ALL MOFs in the dataset (no fine-tuning — raw pretrained features).
#       Output: embeddings_pretrained.npz
#
#   1b. Creates embedding-informed train/val/test splits using cosine similarity
#       in the 768-dim space. Strategy D (farthest-point coverage) ensures every
#       val/test positive has a structurally similar training positive.
#       Output: splits/strategy_d_farthest_point/{train,val,test}_bandgaps_regression.json
#
# PREREQUISITES:
#   - MOF structure files under data/raw/{train,val,test}/ (or test-only pool; see data/README.md)
#   - Bandgap JSONs: data/raw/{train,val,test}_bandgaps_regression.json (at least test_* for a single-pool layout)
#   - moftransformer installed with pretrained weights
#
# USAGE:
#   sbatch scripts/01_extract_embeddings.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$DATA_DIR/embeddings" "$DATA_DIR/splits"

# ---- Step 1a: Extract pretrained embeddings ---------------------------------
section "STEP 1a: EXTRACT PRETRAINED EMBEDDINGS"

python data_preparation/analyze_embeddings.py \
    --data_dir "$DATA_DIR/raw" \
    --output_dir "$DATA_DIR/embeddings"

echo "  Output: $DATA_DIR/embeddings/embeddings_pretrained.npz"

# ---- Step 1b: Create embedding-informed splits ------------------------------
section "STEP 1b: CREATE EMBEDDING-INFORMED SPLITS (Strategy D)"

# The repository ships the exact split used in the paper at
# data/splits/strategy_d_farthest_point/. Re-running this step generates a
# FRESH split (intended for new datasets or new target properties) and would
# overwrite those files, so any existing split JSONs are backed up first.
if ls "$DATA_DIR/splits/strategy_d_farthest_point/"*_bandgaps_regression.json >/dev/null 2>&1; then
    BACKUP_DIR="$DATA_DIR/splits/strategy_d_farthest_point/backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp "$DATA_DIR/splits/strategy_d_farthest_point/"*_bandgaps_regression.json "$BACKUP_DIR/"
    echo "  WARNING: existing split JSONs (e.g. the released paper split) backed up to:"
    echo "           $BACKUP_DIR"
fi

python data_preparation/embedding_split.py \
    --embeddings_path "$DATA_DIR/embeddings/embeddings_pretrained.npz" \
    --splits_dir "$DATA_DIR/raw" \
    --data_dir "$DATA_DIR/raw" \
    --output_dir "$DATA_DIR/splits" \
    --strategy D

echo "  Output: $DATA_DIR/splits/strategy_d_farthest_point/"

# ---- Step 1c: Fix symlinks (if structure files use symlinks) ----------------
section "STEP 1c: REPAIR SPLIT SYMLINKS"

python data_preparation/repair_split_symlinks.py \
    --splits_dir "$DATA_DIR/splits/strategy_d_farthest_point" \
    --data_dir "$DATA_DIR/raw"

section "STEP 1 COMPLETE"
echo "  Embeddings: $DATA_DIR/embeddings/embeddings_pretrained.npz"
echo "  Splits:     $DATA_DIR/splits/strategy_d_farthest_point/"
