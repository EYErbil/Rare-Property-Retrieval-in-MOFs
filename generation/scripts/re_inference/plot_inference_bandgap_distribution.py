#!/usr/bin/env python3
"""
Plot distribution of inferred bandgaps from inference_predictions.csv.

Expected CSV columns:
    cif_id,score,predicted_binary,true_label,mode

Usage:
  python scripts/re_infer/plot_inference_bandgap_distribution.py \
      --csv /path/to/inference_predictions.csv \
      --out /path/to/bandgap_distribution.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_scores(csv_path: Path) -> np.ndarray:
    scores: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "score" not in (reader.fieldnames or []):
            raise ValueError(f"'score' column not found in: {csv_path}")
        for row in reader:
            raw = (row.get("score") or "").strip()
            if not raw:
                continue
            try:
                scores.append(float(raw))
            except ValueError:
                continue
    if not scores:
        raise ValueError(f"No valid numeric scores found in: {csv_path}")
    return np.asarray(scores, dtype=np.float64)


def compute_smoothed_hist_curve(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Lightweight smoothing without extra dependencies.
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    smoothed = np.convolve(hist, kernel, mode="same")
    return centers, smoothed


def make_plot(scores: np.ndarray, out_path: Path, bins: int, title: str | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax_hist, ax_cdf = axes

    # Histogram + smoothed density-like curve
    ax_hist.hist(
        scores,
        bins=bins,
        color="#4C78A8",
        alpha=0.45,
        edgecolor="black",
        linewidth=0.3,
        density=True,
        label="Histogram (density)",
    )
    x_smooth, y_smooth = compute_smoothed_hist_curve(scores, bins=bins)
    ax_hist.plot(x_smooth, y_smooth, color="#E45756", linewidth=2.0, label="Smoothed curve")

    mean_v = float(np.mean(scores))
    med_v = float(np.median(scores))
    ax_hist.axvline(mean_v, color="#1B9E77", linestyle="--", linewidth=1.5, label=f"Mean: {mean_v:.3f}")
    ax_hist.axvline(med_v, color="#7F3C8D", linestyle=":", linewidth=1.8, label=f"Median: {med_v:.3f}")
    ax_hist.set_xlabel("Predicted bandgap (score)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Bandgap Distribution")
    ax_hist.legend(frameon=False, fontsize=8)

    # Empirical CDF
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1, dtype=np.float64) / len(sorted_scores)
    ax_cdf.plot(sorted_scores, cdf, color="#4C78A8", linewidth=2.0)
    ax_cdf.set_xlabel("Predicted bandgap (score)")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_title("Empirical CDF")
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.grid(alpha=0.25, linewidth=0.5)

    p10, p50, p90 = np.percentile(scores, [10, 50, 90])
    stats_text = (
        f"N = {len(scores)}\n"
        f"min = {scores.min():.3f}\n"
        f"p10 = {p10:.3f}\n"
        f"p50 = {p50:.3f}\n"
        f"p90 = {p90:.3f}\n"
        f"max = {scores.max():.3f}"
    )
    ax_cdf.text(
        0.98,
        0.02,
        stats_text,
        transform=ax_cdf.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.suptitle(title or "Inference Bandgap Score Distribution", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot inferred bandgap score distribution from inference_predictions.csv.")
    p.add_argument("--csv", type=Path, required=True, help="Path to inference_predictions.csv")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("inference_bandgap_distribution.png"),
        help="Output figure path (default: inference_bandgap_distribution.png)",
    )
    p.add_argument("--bins", type=int, default=60, help="Number of histogram bins (default: 60)")
    p.add_argument("--title", type=str, default=None, help="Optional figure title")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    scores = load_scores(args.csv)
    make_plot(scores, args.out, bins=max(10, args.bins), title=args.title)
    print(f"Saved plot -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
