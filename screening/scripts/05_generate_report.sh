#!/bin/bash
#SBATCH --job-name=report
#SBATCH --output=logs/05_report_%j.out
#SBATCH --error=logs/05_report_%j.err
#SBATCH --partition=mid
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# =============================================================================
# STEP 5: Generate Comprehensive Analysis Report
# =============================================================================
#
# Reads all ensemble results and produces 15+ analysis figures:
#   - Model leaderboard (ensemble vs individual)
#   - Confusion matrices (top-K discovery)
#   - Recall/precision curves per model
#   - Complementarity analysis (which MOFs each model uniquely finds)
#   - Metal center and bandgap distribution breakdowns
#   - Top-20 MOF candidate lists with metadata
#   - UMAP visualizations (if available)
#
# OUTPUT:
#   final_results/
#     ├── summary.md                    — Text report
#     ├── recommended_combinations.csv  — Best model combos
#     ├── fig01_leaderboard.png         — Model comparison
#     ├── fig02_confusion.png           — Discovery confusion matrix
#     └── ... (15+ figures)
#
# PREREQUISITES:
#   - Step 04 completed (ensemble_results/ populated)
#   - Optional: qmof.csv in data/ for metadata enrichment
#
# USAGE:
#   sbatch scripts/05_generate_report.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
load_modules
cd "$BASE_DIR"
mkdir -p logs "$REPORT_DIR"

section "GENERATING FINAL REPORT"

python src/generate_final_report.py \
    --base_dir "$BASE_DIR" \
    --output_dir "$REPORT_DIR" \
    --ensemble_dir "$ENSEMBLE_DIR"

section "STEP 5 COMPLETE — FINAL REPORT"
echo "  Output: $REPORT_DIR/"
echo "  Figures: $(ls "$REPORT_DIR"/*.png 2>/dev/null | wc -l) PNG files"
