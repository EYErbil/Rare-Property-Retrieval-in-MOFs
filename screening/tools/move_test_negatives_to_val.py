#!/usr/bin/env python3
"""
Move 400 test negatives to validation (stratified over bandgap 1.5–7 eV).
=======================================================================
Updates all split JSONs in place and writes test_to_val_cifs.txt for
fix_split_symlinks.sh to move symlinks (test → val).

Only NEGATIVES are moved (bandgap >= 1.0); no positives touched.
Selection is stratified so the 400 are spread over 1.5–7 eV.

Usage:
  python move_test_negatives_to_val.py --splits_dir ./new_splits/strategy_d_farthest_point --n_move 400

  # Then on cluster, move symlinks:
  bash fix_split_symlinks.sh
"""

import os
import sys
import json
import argparse
import numpy as np

# Reuse label generation from resplit_data
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


def select_by_interval_counts(test_bandgaps, interval_specs, seed=42):
    """
    Select CIFs from test with fixed counts per interval.
    interval_specs: list of (lo, hi, n) e.g. [(3.0, 3.5, 300), (3.5, 4.0, 400), (4.0, 4.5, 300)]
    Returns flat list of cif_ids.
    """
    rng = np.random.default_rng(seed)
    selected = []
    for lo, hi, n in interval_specs:
        candidates = [cid for cid, bg in test_bandgaps.items() if lo <= bg < hi]
        if len(candidates) < n:
            raise ValueError(
                f"Only {len(candidates)} test CIFs in [{lo}, {hi}) eV; need {n}. "
                f"Reduce n or widen interval."
            )
        take = rng.choice(candidates, size=n, replace=False)
        selected.extend(take.tolist())
    return selected


def select_stratified_negatives(test_bandgaps, n_move, bg_min=1.5, bg_max=7.0, n_bins=10, seed=42):
    """
    Select n_move test CIFs that are negatives (bg >= 1.0) with bandgap in [bg_min, bg_max],
    stratified across n_bins so the distribution is spread over the range.
    """
    rng = np.random.default_rng(seed)
    # Restrict to negatives in range
    candidates = [
        (cid, bg) for cid, bg in test_bandgaps.items()
        if bg >= 1.0 and bg_min <= bg <= bg_max
    ]
    if len(candidates) < n_move:
        raise ValueError(
            f"Only {len(candidates)} test negatives in [{bg_min}, {bg_max}] eV; "
            f"need at least {n_move}. Reduce n_move or widen range."
        )

    # Bin edges 1.5 to 7.0
    edges = np.linspace(bg_min, bg_max, n_bins + 1)
    bins = [[] for _ in range(n_bins)]
    for cid, bg in candidates:
        idx = np.searchsorted(edges[1:], bg)  # which bin (0 .. n_bins-1)
        idx = min(idx, n_bins - 1)
        bins[idx].append((cid, bg))

    # Sample proportionally from each bin to get n_move total
    total_in_bins = sum(len(b) for b in bins)
    selected = []
    for b in bins:
        if not b:
            continue
        # Target count for this bin (proportional)
        target = max(1, round(n_move * len(b) / total_in_bins))
        take = min(target, len(b))
        indices = rng.choice(len(b), size=take, replace=False)
        for i in indices:
            selected.append(b[i][0])

    # If we're short (rounding), add from largest bins until we have n_move
    by_bin_size = sorted(range(n_bins), key=lambda i: len(bins[i]), reverse=True)
    already = set(selected)
    for idx in by_bin_size:
        if len(selected) >= n_move:
            break
        for cid, bg in bins[idx]:
            if cid in already:
                continue
            selected.append(cid)
            already.add(cid)
            if len(selected) >= n_move:
                break

    # If we have too many (rounding up), drop randomly
    if len(selected) > n_move:
        selected = rng.choice(selected, size=n_move, replace=False).tolist()

    return selected[:n_move]


def main():
    ap = argparse.ArgumentParser(description="Move 400 test negatives to val (stratified 1.5–7 eV)")
    ap.add_argument("--splits_dir", type=str, default="./new_splits/strategy_d_farthest_point",
                    help="Split directory (train/val/test_bandgaps_regression.json)")
    ap.add_argument("--n_move", type=int, default=None,
                    help="Number of test negatives to move to val (stratified). Ignored if --interval_counts given.")
    ap.add_argument("--interval_counts", type=str, default=None,
                    help="Fixed counts per interval, e.g. '3.0-3.5:300,3.5-4.0:400,4.0-4.5:300'")
    ap.add_argument("--bg_min", type=float, default=1.5, help="Min bandgap (eV) for selection range (used only with --n_move)")
    ap.add_argument("--bg_max", type=float, default=7.0, help="Max bandgap (eV) for selection range")
    ap.add_argument("--n_bins", type=int, default=10, help="Number of bins for stratification")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits_dir = os.path.abspath(args.splits_dir)
    if not os.path.isdir(splits_dir):
        print(f"ERROR: {splits_dir} not found.")
        sys.exit(1)

    print("Loading test and val splits...")
    test_bg = load_split_json(splits_dir, "test")
    val_bg = load_split_json(splits_dir, "val")

    n_test_before = len(test_bg)
    n_val_before = len(val_bg)
    n_test_pos = sum(1 for bg in test_bg.values() if bg < 1.0)
    n_val_pos = sum(1 for bg in val_bg.values() if bg < 1.0)

    print(f"  Test: {n_test_before} total, {n_test_pos} positives")
    print(f"  Val:  {n_val_before} total, {n_val_pos} positives")

    if args.interval_counts:
        # Parse "3.0-3.5:300,3.5-4.0:400,4.0-4.5:300"
        specs = []
        for part in args.interval_counts.split(","):
            part = part.strip()
            if ":" not in part or "-" not in part:
                raise ValueError(f"Invalid interval_counts part: {part}. Use format lo-hi:n e.g. 3.0-3.5:300")
            range_part, n_str = part.rsplit(":", 1)
            lo_str, hi_str = range_part.split("-", 1)
            specs.append((float(lo_str), float(hi_str), int(n_str)))
        print(f"\n  Interval counts: {specs}")
        move_cids = select_by_interval_counts(test_bg, specs, seed=args.seed)
    else:
        n_move = args.n_move if args.n_move is not None else 400
        move_cids = select_stratified_negatives(
            test_bg, n_move, bg_min=args.bg_min, bg_max=args.bg_max,
            n_bins=args.n_bins, seed=args.seed,
        )
    move_set = set(move_cids)

    # Bandgaps of moved CIFs (for adding to val)
    move_bandgaps = {cid: test_bg[cid] for cid in move_cids}

    # New test: remove moved
    test_bg_new = {cid: bg for cid, bg in test_bg.items() if cid not in move_set}
    # New val: add moved
    val_bg_new = {**val_bg, **move_bandgaps}

    mode = "interval_counts" if args.interval_counts else f"stratified {args.bg_min}-{args.bg_max} eV"
    print(f"\nMoving {len(move_cids)} test negatives ({mode}) to val")
    print(f"  Test: {n_test_before} -> {len(test_bg_new)}")
    print(f"  Val:  {n_val_before} -> {len(val_bg_new)}")

    # Write updated regression JSONs
    save_split_json(test_bg_new, splits_dir, "test")
    save_split_json(val_bg_new, splits_dir, "val")

    # Regenerate all label variants for test and val (binary, ordinal, multiclass)
    print("\nRegenerating label files (binary, ordinal, multiclass)...")
    generate_all_labels(test_bg_new, "test", splits_dir)
    generate_all_labels(val_bg_new, "val", splits_dir)
    # Train unchanged; no need to regenerate train (or its weights)

    # Write list for fix_split_symlinks.sh
    list_path = os.path.join(splits_dir, "test_to_val_cifs.txt")
    with open(list_path, "w") as f:
        for cid in move_cids:
            f.write(cid + "\n")
    print(f"\nWrote {list_path} ({len(move_cids)} CIFs)")

    print("\nDone. Next on cluster: run  bash fix_split_symlinks.sh")
    print("  (fix_split_symlinks.sh will move symlinks for these CIFs from test/ to val/)")
    print("\n--- After split change: clear outdated results and rerun ---")
    print("  See AFTER_SPLIT_MOVE.md for full list of commands.")
    print("\nSee AFTER_SPLIT_CHANGE.md for what to clear and rerun after the split change.")


if __name__ == "__main__":
    main()
