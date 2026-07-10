#!/bin/bash
# Fix symlinks after swapping:
#   BADXEJ01_FSR  val  → test
#   ILEDIO_FSR    test → train
#   HISGEX_FSR    val  → train
#
# Run ONCE from the project root on the cluster:
#   bash fix_split_symlinks.sh
#
# If NN training fails with FileNotFoundError for a test CIF (e.g. PUPXII_FSR.grid),
# run repair_split_symlinks.py first to create any missing symlinks from new_subset:
#   python repair_split_symlinks.py --splits_dir .../new_splits/strategy_d_farthest_point --source_dir .../new_subset

SPLIT_DIR="${SPLIT_DIR:-./data/splits/strategy_d_farthest_point}"
DATA_SRC="${DATA_SRC:-./data/raw}"

echo "=== Fixing symlinks for split swap ==="
echo "  BADXEJ01_FSR: val → test"
echo "  ILEDIO_FSR:   test → train"
echo "  HISGEX_FSR:   val → train"
echo ""

move_symlinks() {
    local cif_id="$1"
    local from_split="$2"
    local to_split="$3"

    mkdir -p "$SPLIT_DIR/$to_split"

    for ext in grid griddata16 graphdata; do
        src="$SPLIT_DIR/$from_split/${cif_id}.$ext"
        dst="$SPLIT_DIR/$to_split/${cif_id}.$ext"

        if [ -e "$src" ]; then
            # If it's a symlink, read the real target and re-link
            if [ -L "$src" ]; then
                real_target=$(readlink -f "$src")
                rm "$src"
                ln -s "$real_target" "$dst"
                echo "  $cif_id.$ext: $from_split → $to_split (relinked to $real_target)"
            else
                mv "$src" "$dst"
                echo "  $cif_id.$ext: $from_split → $to_split (moved)"
            fi
        else
            echo "  WARNING: $src not found, trying to create from source..."
            # Try to find in DATA_SRC
            for orig_split in train val test; do
                orig_src="$DATA_SRC/$orig_split/${cif_id}.$ext"
                if [ -f "$orig_src" ]; then
                    ln -s "$orig_src" "$dst"
                    echo "  $cif_id.$ext: linked from $orig_src → $to_split"
                    break
                fi
            done
        fi
    done
}

# Move BADXEJ01_FSR: val → test
move_symlinks "BADXEJ01_FSR" "val" "test"

echo ""

# Move ILEDIO_FSR: test → train
move_symlinks "ILEDIO_FSR" "test" "train"

echo ""

# Move HISGEX_FSR: val → train (UMAP bottom region, x=6.54, y=-5.90, Rb, bg=0.1129)
move_symlinks "HISGEX_FSR" "val" "train"

echo ""
echo "=== Move test negatives → val (from test_to_val_cifs.txt) ==="
TEST_TO_VAL_LIST="$SPLIT_DIR/test_to_val_cifs.txt"
if [ -f "$TEST_TO_VAL_LIST" ]; then
    n=0
    while IFS= read -r cif_id || [ -n "$cif_id" ]; do
        [ -z "$cif_id" ] && continue
        move_symlinks "$cif_id" "test" "val"
        n=$((n + 1))
    done < "$TEST_TO_VAL_LIST"
    echo "  Moved $n CIFs from test/ to val/"
else
    echo "  (No $TEST_TO_VAL_LIST — run move_test_negatives_to_val.py first to create it)"
fi

echo ""
echo "=== Move val → test (from val_to_test_cifs.txt) ==="
VAL_TO_TEST_LIST="$SPLIT_DIR/val_to_test_cifs.txt"
if [ -f "$VAL_TO_TEST_LIST" ]; then
    n=0
    while IFS= read -r cif_id || [ -n "$cif_id" ]; do
        [ -z "$cif_id" ] && continue
        move_symlinks "$cif_id" "val" "test"
        n=$((n + 1))
    done < "$VAL_TO_TEST_LIST"
    echo "  Moved $n CIFs from val/ to test/"
else
    echo "  (No $VAL_TO_TEST_LIST — run move_val_to_test.py first to create it)"
fi

echo ""
echo "=== Verification ==="
echo "BADXEJ01_FSR in test/:"
ls -la "$SPLIT_DIR/test/BADXEJ01_FSR"* 2>/dev/null || echo "  NOT FOUND"
echo "BADXEJ01_FSR in val/ (should be gone):"
ls -la "$SPLIT_DIR/val/BADXEJ01_FSR"* 2>/dev/null || echo "  Gone (correct)"
echo ""
echo "ILEDIO_FSR in train/:"
ls -la "$SPLIT_DIR/train/ILEDIO_FSR"* 2>/dev/null || echo "  NOT FOUND"
echo "ILEDIO_FSR in test/ (should be gone):"
ls -la "$SPLIT_DIR/test/ILEDIO_FSR"* 2>/dev/null || echo "  Gone (correct)"
echo ""
echo "HISGEX_FSR in train/:"
ls -la "$SPLIT_DIR/train/HISGEX_FSR"* 2>/dev/null || echo "  NOT FOUND"
echo "HISGEX_FSR in val/ (should be gone):"
ls -la "$SPLIT_DIR/val/HISGEX_FSR"* 2>/dev/null || echo "  Gone (correct)"
echo ""
echo "Done! Now run reinfer_nn.py to update NN predictions."
