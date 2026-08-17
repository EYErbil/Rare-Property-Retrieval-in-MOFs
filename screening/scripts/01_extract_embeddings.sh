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
# STEP 1: Restore the paper split, or explicitly create a fresh split
# =============================================================================
#
# Default (SPLIT_MODE=paper):
#   Restore/verify the exact pre-training PMTransformer partition recorded in
#   the labeled embedding archive. Existing canonical JSONs are never replaced.
#
# Explicit opt-in (SPLIT_MODE=fresh):
#   Extract new pretrained embeddings and design a new split under
#   data/{embeddings,splits}/noncanonical/<run-id>/. This path is for a new
#   dataset or target and cannot overwrite the paper partition.
#
# Examples:
#   sbatch scripts/01_extract_embeddings.sh
#   SPLIT_MODE=fresh FRESH_RUN_ID=my_new_target sbatch scripts/01_extract_embeddings.sh
#
# Optional canonical archive override:
#   PAPER_EMBEDDINGS_ARCHIVE=/downloaded/pmt_embeddings_qmof_labeled.npz
# Optional source/preprocessing override:
#   SPLIT_SOURCE_DIR=/path/to/preprocessed/qmof
# Set REPAIR_SPLIT_LINKS=0 only to verify/materialize JSON membership without
# preparing the train/validation/test structure links needed by Step 2.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"

SPLIT_MODE="${SPLIT_MODE:-paper}"
CANONICAL_SPLIT_DIR="$DATA_DIR/splits/strategy_d_farthest_point"
PAPER_EMBEDDINGS_ARCHIVE="${PAPER_EMBEDDINGS_ARCHIVE:-$DATA_DIR/embeddings/embeddings_pretrained.npz}"
SPLIT_SOURCE_DIR="${SPLIT_SOURCE_DIR:-$DATA_DIR/raw}"
REPAIR_SPLIT_LINKS="${REPAIR_SPLIT_LINKS:-1}"

case "$SPLIT_MODE" in
    paper)
        section "STEP 1: MATERIALIZE/VERIFY EXACT PAPER SPLIT"

        if [[ ! -f "$PAPER_EMBEDDINGS_ARCHIVE" ]]; then
            echo "ERROR: labeled PMTransformer archive not found:"
            echo "       $PAPER_EMBEDDINGS_ARCHIVE"
            echo "       Existing canonical split JSONs were left untouched."
            echo "       Restore the Zenodo archive there or set PAPER_EMBEDDINGS_ARCHIVE."
            exit 2
        fi

        python data_preparation/materialize_paper_split.py \
            --archive "$PAPER_EMBEDDINGS_ARCHIVE" \
            --output-dir "$CANONICAL_SPLIT_DIR"

        SELECTED_SPLIT_DIR="$CANONICAL_SPLIT_DIR"
        ;;

    fresh)
        FRESH_RUN_ID="${FRESH_RUN_ID:-fresh_$(date +%Y%m%d_%H%M%S)}"
        if [[ ! "$FRESH_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
            echo "ERROR: FRESH_RUN_ID may contain only letters, digits, '.', '_', and '-'."
            exit 2
        fi

        FRESH_EMBEDDING_DIR="$DATA_DIR/embeddings/noncanonical/$FRESH_RUN_ID"
        FRESH_SPLIT_ROOT="$DATA_DIR/splits/noncanonical/$FRESH_RUN_ID"
        if [[ -e "$FRESH_EMBEDDING_DIR" || -e "$FRESH_SPLIT_ROOT" ]]; then
            echo "ERROR: noncanonical run '$FRESH_RUN_ID' already exists; choose a new FRESH_RUN_ID."
            exit 2
        fi

        mkdir -p "$FRESH_EMBEDDING_DIR" "$FRESH_SPLIT_ROOT" logs

        section "STEP 1a: EXTRACT FRESH PRETRAINED EMBEDDINGS (NONCANONICAL)"
        python data_preparation/analyze_embeddings.py \
            --data_dir "$DATA_DIR/raw" \
            --output_dir "$FRESH_EMBEDDING_DIR"

        FRESH_EMBEDDINGS_ARCHIVE="$FRESH_EMBEDDING_DIR/embeddings_pretrained.npz"
        if [[ ! -f "$FRESH_EMBEDDINGS_ARCHIVE" ]]; then
            echo "ERROR: embedding extraction did not create $FRESH_EMBEDDINGS_ARCHIVE"
            exit 2
        fi

        section "STEP 1b: DESIGN FRESH PMTRANSFORMER SPLIT (NONCANONICAL)"
        python data_preparation/embedding_split.py \
            --embeddings_path "$FRESH_EMBEDDINGS_ARCHIVE" \
            --splits_dir "$DATA_DIR/raw" \
            --data_dir "$DATA_DIR/raw" \
            --output_dir "$FRESH_SPLIT_ROOT" \
            --strategy D

        SELECTED_SPLIT_DIR="$FRESH_SPLIT_ROOT/strategy_d_farthest_point"
        ;;

    *)
        echo "ERROR: SPLIT_MODE must be 'paper' (default) or 'fresh'."
        exit 2
        ;;
esac

if [[ "$REPAIR_SPLIT_LINKS" == "1" ]]; then
    section "STEP 1c: REPAIR SPLIT SYMLINKS"
    python data_preparation/repair_split_symlinks.py \
        --splits_dir "$SELECTED_SPLIT_DIR" \
        --source_dir "$SPLIT_SOURCE_DIR"
elif [[ "$REPAIR_SPLIT_LINKS" != "0" ]]; then
    echo "ERROR: REPAIR_SPLIT_LINKS must be 0 or 1."
    exit 2
fi

section "STEP 1 COMPLETE"
if [[ "$SPLIT_MODE" == "paper" ]]; then
    echo "  Archive: $PAPER_EMBEDDINGS_ARCHIVE"
else
    echo "  Archive: $FRESH_EMBEDDINGS_ARCHIVE"
fi
echo "  Splits:  $SELECTED_SPLIT_DIR"
