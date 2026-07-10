#!/usr/bin/env python3
"""
Report bandgap distribution by split (train/val/test) in 0.5 eV intervals.

Loads *_bandgaps_regression.json from a split directory and prints/plots
counts per 0.5 eV bin (0–0.5, 0.5–1.0, 1.0–1.5, ...).

Usage:
  python report_split_bandgap_distribution.py
  python report_split_bandgap_distribution.py --splits_dir ./new_splits/strategy_d_farthest_point
  python report_split_bandgap_distribution.py --splits_dir ./new_splits/strategy_d_farthest_point --max_bg 8 --output_dir ./reports
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_split(splits_dir, split_name):
    path = os.path.join(splits_dir, f"{split_name}_bandgaps_regression.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {cid: float(bg) for cid, bg in data.items()}


def bin_counts(bandgaps, bin_width=0.5, max_bg=8.0):
    """Return counts per bin. Bins: [0, 0.5), [0.5, 1.0), ..., [max_bg-bin_width, max_bg)."""
    edges = np.arange(0, max_bg + 1e-9, bin_width)
    n_bins = len(edges) - 1
    counts = np.zeros(n_bins)
    for bg in bandgaps.values():
        if bg < 0:
            idx = 0
        elif bg >= max_bg:
            idx = n_bins - 1
        else:
            idx = min(int(bg / bin_width), n_bins - 1)
        counts[idx] += 1
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(n_bins)]
    return counts, labels


def main():
    ap = argparse.ArgumentParser(description="Report bandgap distribution by split in 0.5 eV bins")
    ap.add_argument("--splits_dir", type=str, default="./new_splits/strategy_d_farthest_point",
                    help="Split directory with train/val/test_bandgaps_regression.json")
    ap.add_argument("--bin_width", type=float, default=0.5, help="Bin width (eV)")
    ap.add_argument("--max_bg", type=float, default=8.0, help="Upper bandgap limit (eV)")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Directory for plot (default: splits_dir)")
    args = ap.parse_args()

    splits_dir = os.path.abspath(args.splits_dir)
    if not os.path.isdir(splits_dir):
        print(f"ERROR: {splits_dir} not found.")
        sys.exit(1)

    output_dir = args.output_dir or splits_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load splits
    train = load_split(splits_dir, "train")
    val = load_split(splits_dir, "val")
    test = load_split(splits_dir, "test")

    # Bin counts
    train_counts, labels = bin_counts(train, args.bin_width, args.max_bg)
    val_counts, _ = bin_counts(val, args.bin_width, args.max_bg)
    test_counts, _ = bin_counts(test, args.bin_width, args.max_bg)

    n_bins = len(labels)

    # --- Print report ---
    print("=" * 70)
    print("  Bandgap distribution by split (bin width = {:.1f} eV, max = {:.1f} eV)".format(
        args.bin_width, args.max_bg))
    print("=" * 70)
    print(f"  Split dir: {splits_dir}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("=" * 70)
    print(f"  {'Bin (eV)':<14}  {'Train':>8}  {'Val':>8}  {'Test':>8}  {'Total':>8}")
    print("-" * 70)
    for i in range(n_bins):
        t, v, te = int(train_counts[i]), int(val_counts[i]), int(test_counts[i])
        total = t + v + te
        print(f"  {labels[i]:<14}  {t:>8}  {v:>8}  {te:>8}  {total:>8}")
    print("-" * 70)
    print(f"  {'Total':<14}  {int(train_counts.sum()):>8}  {int(val_counts.sum()):>8}  "
          f"{int(test_counts.sum()):>8}  {int(train_counts.sum() + val_counts.sum() + test_counts.sum()):>8}")
    print("=" * 70)

    # --- Plot ---
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_bins)
        w = 0.25
        ax.bar(x - w, train_counts, width=w, label="Train", color="tab:blue")
        ax.bar(x, val_counts, width=w, label="Val", color="tab:orange")
        ax.bar(x + w, test_counts, width=w, label="Test", color="tab:green")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Bandgap (eV)")
        ax.set_ylabel("Count")
        ax.set_title(f"Bandgap distribution by split — {os.path.basename(splits_dir)}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "split_bandgap_distribution.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\n  Plot saved: {out_path}")
    else:
        print("\n  (matplotlib not available — skipping plot)")

    print()


if __name__ == "__main__":
    main()
