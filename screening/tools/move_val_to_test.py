#!/usr/bin/env python3
"""
Move N val structures to test (from a bandgap interval).
=======================================================================
Updates all split JSONs in place and writes val_to_test_cifs.txt for
fix_split_symlinks.sh to move symlinks (val → test).

Usage:
  python move_val_to_test.py --splits_dir ./new_splits/strategy_d_farthest_point \\
    --bg_min 1.0 --bg_max 1.5 --n_move 3

  # Then on cluster, move symlinks:
  bash fix_split_symlinks.sh
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resplit_data import generate_all_labels


def load_split_json(splits_dir, split_name):
    path = os.path.join(splits_dir, f"{split_name}_bandgaps_regression.json")
    with open(path) as f:
        data = json.load(f)
    return {cid: float(bg) for cid, bg in data.items()}


def save_split_json(data, splits_dir, split_name):
    path = os.path.join(splits_dir, f"{split_name}_bandgaps_regression.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Move N val structures to test (from bandgap interval)")
    ap.add_argument("--splits_dir", type=str, default="./new_splits/strategy_d_farthest_point",
                    help="Split directory (train/val/test_bandgaps_regression.json)")
    ap.add_argument("--bg_min", type=float, default=1.0, help="Min bandgap (eV) for selection [inclusive]")
    ap.add_argument("--bg_max", type=float, default=1.5, help="Max bandgap (eV) for selection [exclusive]")
    ap.add_argument("--n_move", type=int, default=3, help="Number of val structures to move to test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits_dir = os.path.abspath(args.splits_dir)
    if not os.path.isdir(splits_dir):
        print(f"ERROR: {splits_dir} not found.")
        sys.exit(1)

    print("Loading val and test splits...")
    val_bg = load_split_json(splits_dir, "val")
    test_bg = load_split_json(splits_dir, "test")

    candidates = [
        cid for cid, bg in val_bg.items()
        if args.bg_min <= bg < args.bg_max
    ]

    if len(candidates) < args.n_move:
        print(f"ERROR: Only {len(candidates)} val structures in [{args.bg_min}, {args.bg_max}) eV; need {args.n_move}.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    move_cids = rng.choice(candidates, size=args.n_move, replace=False).tolist()
    move_set = set(move_cids)
    move_bandgaps = {cid: val_bg[cid] for cid in move_cids}

    n_val_before = len(val_bg)
    n_test_before = len(test_bg)
    val_bg_new = {cid: bg for cid, bg in val_bg.items() if cid not in move_set}
    test_bg_new = {**test_bg, **move_bandgaps}

    print(f"\nMoving {len(move_cids)} val structures (bandgap [{args.bg_min}, {args.bg_max}) eV) to test")
    print(f"  Val:  {n_val_before} -> {len(val_bg_new)}")
    print(f"  Test: {n_test_before} -> {len(test_bg_new)}")
    for cid in move_cids:
        print(f"    {cid}  {val_bg[cid]:.4f} eV")

    save_split_json(val_bg_new, splits_dir, "val")
    save_split_json(test_bg_new, splits_dir, "test")

    print("\nRegenerating label files (binary, ordinal, multiclass)...")
    generate_all_labels(val_bg_new, "val", splits_dir)
    generate_all_labels(test_bg_new, "test", splits_dir)

    list_path = os.path.join(splits_dir, "val_to_test_cifs.txt")
    with open(list_path, "w") as f:
        for cid in move_cids:
            f.write(cid + "\n")
    print(f"\nWrote {list_path} ({len(move_cids)} CIFs)")

    print("\nDone. Next on cluster: run  bash fix_split_symlinks.sh")
    print("  (fix_split_symlinks.sh will move symlinks for these CIFs from val/ to test/)")


if __name__ == "__main__":
    main()
