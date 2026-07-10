#!/bin/bash
# =============================================================================
# Collect MOF Structure Files for Unlabeled Inference
# =============================================================================
#
# This utility copies the three required MOFTransformer input files
# (.graphdata, .grid, .griddata16) for each MOF listed in a JSON file
# into a single directory suitable for inference.
#
# WHY IS THIS NEEDED?
#   MOFTransformer expects structures organized as:
#     data/unlabeled/test/<CIF_ID>.graphdata
#     data/unlabeled/test/<CIF_ID>.grid
#     data/unlabeled/test/<CIF_ID>.griddata16
#
#   But your raw structures may be scattered across multiple directories
#   (e.g., train/val/test splits from a prior experiment). This script
#   searches source directories and gathers everything into one place.
#
# WHAT IT DOES:
#   1. Reads CIF IDs from a JSON file (keys = structure names)
#   2. For each CIF ID, searches SOURCE_DIRS for the 3 required files
#   3. Copies them into TARGET_DIR
#   4. Reports any missing structures
#
# USAGE:
#   # Edit SOURCE_DIRS, TARGET_DIR, and JSON_PATH below, then:
#   bash discovery/collect_inference_structures.sh
#
# =============================================================================

set -euo pipefail

# ---- CONFIGURATION: Edit these paths ----------------------------------------

# Where to look for structure files (space-separated list of directories).
# The script searches ALL of these for each CIF ID.
SOURCE_DIRS="${SOURCE_DIRS:-./data/raw/train ./data/raw/val ./data/raw/test}"

# Where to copy collected files
TARGET_DIR="${TARGET_DIR:-./data/unlabeled/test}"

# JSON file mapping CIF_ID → bandgap (only the keys are used).
# Create this manually or with a helper script:
#   python -c "import json; d = {cif: 0.0 for cif in cif_list}; json.dump(d, open('file.json','w'))"
JSON_PATH="${JSON_PATH:-./data/unlabeled/test_bandgaps_regression.json}"

# ---- Validation -------------------------------------------------------------

if [ ! -f "$JSON_PATH" ]; then
    echo "ERROR: JSON file not found: $JSON_PATH"
    echo ""
    echo "  This file should map CIF IDs to bandgap values (values can be 0.0"
    echo "  for unlabeled structures). Example:"
    echo '    {"QMOF-abc123": 0.0, "QMOF-def456": 0.0}'
    echo ""
    echo "  Create it, then re-run this script."
    exit 1
fi

mkdir -p "$TARGET_DIR"

# ---- Extract CIF IDs from JSON keys ----------------------------------------

echo "Reading structure names from: $JSON_PATH"

STRUCT_NAMES=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for k in sorted(data.keys()):
    print(k)
" "$JSON_PATH")

TOTAL=$(echo "$STRUCT_NAMES" | wc -l)
echo "Total structures to collect: $TOTAL"
echo ""

# ---- Search and copy --------------------------------------------------------

FOUND=0
SKIPPED=0
MISSING=0

for name in $STRUCT_NAMES; do
    # Already collected?
    if [ -f "$TARGET_DIR/${name}.graphdata" ] && \
       [ -f "$TARGET_DIR/${name}.grid" ] && \
       [ -f "$TARGET_DIR/${name}.griddata16" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Search source directories
    graph="" ; grid="" ; grid16=""

    for src_dir in $SOURCE_DIRS; do
        [ -z "$graph"  ] && [ -f "$src_dir/${name}.graphdata" ]  && graph="$src_dir/${name}.graphdata"
        [ -z "$grid"   ] && [ -f "$src_dir/${name}.grid" ]       && grid="$src_dir/${name}.grid"
        [ -z "$grid16" ] && [ -f "$src_dir/${name}.griddata16" ] && grid16="$src_dir/${name}.griddata16"
    done

    if [ -n "$graph" ] && [ -n "$grid" ] && [ -n "$grid16" ]; then
        cp -n "$graph"  "$TARGET_DIR/${name}.graphdata"
        cp -n "$grid"   "$TARGET_DIR/${name}.grid"
        cp -n "$grid16" "$TARGET_DIR/${name}.griddata16"
        FOUND=$((FOUND + 1))
    else
        echo "MISSING: $name"
        [ -z "$graph" ]  && echo "  - ${name}.graphdata not found"
        [ -z "$grid" ]   && echo "  - ${name}.grid not found"
        [ -z "$grid16" ] && echo "  - ${name}.griddata16 not found"
        MISSING=$((MISSING + 1))
    fi
done

# ---- Summary ----------------------------------------------------------------

echo ""
echo "=========================================="
echo "  Collection complete"
echo "=========================================="
echo "  Total in JSON:    $TOTAL"
echo "  Already present:  $SKIPPED"
echo "  Newly copied:     $FOUND"
echo "  Missing:          $MISSING"
echo "  Output directory: $TARGET_DIR"
echo ""

if [ "$MISSING" -gt 0 ]; then
    echo "WARNING: $MISSING structures could not be found in SOURCE_DIRS."
    echo "  Searched: $SOURCE_DIRS"
    echo "  These MOFs will be skipped during inference."
fi
