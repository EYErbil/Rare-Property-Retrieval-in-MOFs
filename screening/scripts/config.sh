#!/bin/bash
# =============================================================================
# CLUSTER CONFIGURATION — Edit these variables for your environment
# =============================================================================

# --- Paths -------------------------------------------------------------------
# The screening/ module directory of the cloned repository (data lives here too).
# All scripts derive paths relative to this.
export BASE_DIR="/path/to/rare-property-retrieval/screening"

# Python virtual environment (must have moftransformer, pytorch-lightning, etc.)
export VENV_PATH="/path/to/venv/bin/activate"

# --- SLURM settings ----------------------------------------------------------
export SLURM_PARTITION_GPU="ai"          # GPU partition name
export SLURM_PARTITION_CPU="mid"         # CPU-only partition name
export SLURM_ACCOUNT="ai"               # Account / QoS (set to "" if not needed)
export SLURM_QOS="ai"

# --- Module loads (cluster-specific) -----------------------------------------
# Adjust to match your module system. Set to "" if not applicable.
export MODULE_LOADS="cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5"

# --- Derived paths (do not edit unless you rearrange the repo) ---------------
export SRC_DIR="$BASE_DIR/src"
export DATA_DIR="$BASE_DIR/data"
export SPLITS_DIR="$DATA_DIR/splits/strategy_d_farthest_point"
export EMB_FILE="$DATA_DIR/embeddings/embeddings_pretrained.npz"
export EXP_BASE="$BASE_DIR/experiments"
export SKLEARN_DIR="$DATA_DIR/embedding_classifiers/strategy_d_farthest_point"
export KNN_DIR="$DATA_DIR/knn_results/strategy_d_farthest_point"
export ENSEMBLE_DIR="$DATA_DIR/ensemble_results"
export REPORT_DIR="$DATA_DIR/final_results"
export DISCOVERY_DATA="$DATA_DIR/unlabeled"
export CIF_DIR="$DATA_DIR/raw/cif"                         # Directory with original .cif files (for SOAP computation in figures/)
export QMOF_CSV="$DATA_DIR/qmof.csv"                       # QMOF metadata (formula, metal centers)
export FIGURES_OUTPUT="$BASE_DIR/figures_output"             # Generated figure outputs

# --- Helper function ---------------------------------------------------------
load_modules() {
    module purge 2>/dev/null || true
    if [ -n "$MODULE_LOADS" ]; then
        module load $MODULE_LOADS
    fi
    source "$VENV_PATH"
}

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

section() {
    echo ""
    echo "=================================================================="
    echo "  $(timestamp) | $1"
    echo "=================================================================="
}
