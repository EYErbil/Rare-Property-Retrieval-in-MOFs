#!/usr/bin/env python3
"""
Repair missing or broken symlinks in a split directory.

The NN dataloader expects data_dir/{split}/{cif_id}.grid (and .griddata16, .graphdata).
If test_bandgaps_regression.json lists a CIF but the file is missing (e.g. broken
symlink or never linked), training/test fails with FileNotFoundError.

This script:
  - Reads train/val/test_bandgaps_regression.json from --splits_dir
  - For each CIF in each split, checks that .grid, .griddata16, .graphdata exist
  - If missing, looks in --source_dir (train/val/test) and creates symlinks

Usage:
  python repair_split_symlinks.py \\
    --splits_dir ./data/splits/strategy_d_farthest_point \\
    --source_dir ./data/raw
"""

import argparse
import json
import os


EXTS = ["grid", "griddata16", "graphdata"]
SPLITS = ["train", "val", "test"]


def main():
    ap = argparse.ArgumentParser(description="Repair missing split symlinks from a source dir")
    ap.add_argument(
        "--splits_dir",
        type=str,
        default="./new_splits/strategy_d_farthest_point",
        help="Split directory (contains train/val/test/ and *_bandgaps_regression.json)",
    )
    ap.add_argument(
        "--source_dir",
        type=str,
        default="./new_subset",
        help="Source data directory (train/val/test/ with actual .grid etc. files)",
    )
    ap.add_argument("--dry_run", action="store_true", help="Only report missing files, do not create links")
    args = ap.parse_args()

    splits_dir = os.path.abspath(args.splits_dir)
    source_dir = os.path.abspath(args.source_dir)
    if not os.path.isdir(splits_dir):
        raise SystemExit(f"ERROR: splits_dir not found: {splits_dir}")
    if not os.path.isdir(source_dir):
        raise SystemExit(f"ERROR: source_dir not found: {source_dir}")

    total_fixed = 0
    total_missing = 0

    for split in SPLITS:
        json_path = os.path.join(splits_dir, f"{split}_bandgaps_regression.json")
        if not os.path.isfile(json_path):
            print(f"  Skip {split}: no {split}_bandgaps_regression.json")
            continue
        with open(json_path, "r") as f:
            cids = list(json.load(f).keys())
        split_dest = os.path.join(splits_dir, split)
        os.makedirs(split_dest, exist_ok=True)

        for cid in cids:
            for ext in EXTS:
                dst = os.path.join(split_dest, f"{cid}.{ext}")
                if os.path.isfile(dst) or (os.path.islink(dst) and os.path.exists(dst)):
                    continue
                # Broken link or missing: resolve from source
                found = None
                for orig in SPLITS:
                    src = os.path.join(source_dir, orig, f"{cid}.{ext}")
                    if os.path.isfile(src):
                        found = src
                        break
                if found:
                    if not args.dry_run:
                        try:
                            os.symlink(found, dst)
                        except OSError as e:
                            print(f"  WARNING: could not link {dst} -> {found}: {e}")
                            total_missing += 1
                            continue
                    total_fixed += 1
                    print(f"  {split}/{cid}.{ext} -> {os.path.relpath(found, splits_dir)}")
                else:
                    total_missing += 1
                    print(f"  MISSING (no source): {split}/{cid}.{ext}")

    if args.dry_run and total_fixed:
        print(f"\nDry run: would create {total_fixed} symlinks.")
    elif total_fixed:
        print(f"\nCreated {total_fixed} symlinks.")
    if total_missing:
        print(f"WARNING: {total_missing} file(s) had no source in {source_dir}. Fix or exclude those CIFs.")
    else:
        print("All CIFs in split JSONs have corresponding files (or were fixed).")


if __name__ == "__main__":
    main()
