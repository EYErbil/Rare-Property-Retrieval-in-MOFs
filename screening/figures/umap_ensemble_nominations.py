#!/usr/bin/env python3
"""
Post-Training Ensemble Nominations UMAP — Fine-Tuned PMTransformer Space
=========================================================================

Same panel as SOAP UMAP panel (d): highlights ensemble top-25 nominations
(red stars) and known positives (blue triangles) on a 2D UMAP, but computed
from fine-tuned PMTransformer embeddings instead of SOAP descriptors.

Generates 3 separate plots — one per experiment (exp364, exp370, exp371).

Usage:
  python figures/umap_ensemble_nominations.py \\
      --npz_exp364 figures_output/finetuned_umap_exp364_fulltune/posttrain_embeddings.npz \\
      --npz_exp370 figures_output/finetuned_umap_exp370_seed2/posttrain_embeddings.npz \\
      --npz_exp371 figures_output/finetuned_umap_exp371_seed3/posttrain_embeddings.npz \\
      --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --nominations data/unlabeled/nomination-SOAP/FINAL_TOP25_diverse.txt \\
      --output_dir figures_output/ensemble_umap

  # Plot-only with cached UMAP coords (fast re-runs):
  python figures/umap_ensemble_nominations.py \\
      --npz_exp364 ... --npz_exp370 ... --npz_exp371 ... \\
      --labeled_splits_dir ... --nominations ... \\
      --output_dir figures_output/ensemble_umap \\
      --load_umap_cache

Requirements: pip install numpy matplotlib umap-learn
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────
BANDGAP_THRESHOLD = 1.0  # eV — positive class boundary


# ──────────────────────────────────────────────────────────────────────
#  Publication style (same as SOAP UMAP script)
# ──────────────────────────────────────────────────────────────────────
def set_publication_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         8,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   6.5,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth":   1.0,
        "patch.linewidth":   0.5,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "mathtext.default":  "regular",
    })


def _save_panel(fig, output_dir, name):
    for fmt in ("png", "svg", "pdf"):
        p = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(p, dpi=600 if fmt == "png" else 300,
                    format=fmt, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    print(f"    Saved {name}.png / .svg / .pdf")


def _style_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=7, labelpad=2)
    ax.set_ylabel("UMAP-2", fontsize=7, labelpad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color("0.5")


# ──────────────────────────────────────────────────────────────────────
#  Flexible ID matching (same as SOAP UMAP script)
# ──────────────────────────────────────────────────────────────────────
def _id_variants(cid):
    """Generate common ID variants for flexible matching (±_FSR, ±.cif)."""
    yield cid
    bare = cid.replace(".cif", "")
    if bare != cid:
        yield bare
    if "_FSR" in bare:
        yield bare.replace("_FSR", "")
    else:
        yield bare + "_FSR"


def _flex_lookup(cid, lookup_dict):
    """Try multiple ID variants to find a match in lookup_dict."""
    for v in _id_variants(cid):
        if v in lookup_dict:
            return lookup_dict[v]
    return None


# ──────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────
def load_npz(npz_path):
    """Load posttrain_embeddings.npz (same format as forward_finetuned_umap.py output)."""
    data = np.load(npz_path, allow_pickle=True)
    cif_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
               for x in data["cif_ids"]]
    embeddings = data["embeddings"]
    bandgaps   = data["bandgaps"]
    splits     = np.array([str(x) for x in data["splits"]])
    is_labeled = data["is_labeled"].astype(bool) if "is_labeled" in data else None
    print(f"    Loaded: {len(cif_ids)} MOFs, dim={embeddings.shape[1]}")
    return cif_ids, embeddings, bandgaps, splits, is_labeled


def load_split_labels(splits_dir):
    """Load bandgap labels and split assignments from split JSONs."""
    if not os.path.isdir(splits_dir):
        print(f"  *** WARNING: labeled_splits_dir does not exist: {splits_dir}")
        return {}, {}
    labels, assignments = {}, {}
    for split_name in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{split_name}_bandgaps_regression.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        for cid, bg in d.items():
            labels[cid] = float(bg)
            assignments[cid] = split_name
        print(f"    {split_name}: {len(d)} MOFs")
    return labels, assignments


def load_nominations(filepath):
    """Load ensemble top-K CIF IDs (one per line)."""
    cids = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                cids.append(line)
    print(f"    Loaded {len(cids)} ensemble nominations")
    return cids


# ──────────────────────────────────────────────────────────────────────
#  UMAP
# ──────────────────────────────────────────────────────────────────────
def compute_umap(embeddings, n_neighbors=30, min_dist=0.3, seed=42):
    try:
        from umap import UMAP
    except ImportError:
        sys.exit("ERROR: umap-learn not installed.  pip install umap-learn")
    print(f"    {embeddings.shape[0]} points, dim={embeddings.shape[1]} ...")
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                   metric="cosine", random_state=seed, n_jobs=-1)
    coords = reducer.fit_transform(embeddings)
    print(f"    UMAP done -> shape {coords.shape}")
    return coords


# ──────────────────────────────────────────────────────────────────────
#  Panel: Ensemble nominations (same as SOAP panel d)
# ──────────────────────────────────────────────────────────────────────
def panel_ensemble_nominations(coords, is_labeled, bandgaps, threshold,
                               nomination_mask, nomination_cids, output_dir,
                               exp_label, filename_prefix):
    """
    Same plot as SOAP UMAP panel (d):
      - Unlabeled MOFs as gray dots, labeled MOFs as blue dots
      - Known positives (bandgap < threshold) as blue triangles
      - Ensemble nominations as red stars with name annotations
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    # Background: unlabeled in gray, labeled in blue
    unlabeled_mask = ~is_labeled
    n_unlab = int(unlabeled_mask.sum())
    n_lab   = int(is_labeled.sum())
    ax.scatter(coords[unlabeled_mask, 0], coords[unlabeled_mask, 1],
               c="#c0c0c0", s=1.5, alpha=0.30, rasterized=True, zorder=1)
    ax.scatter(coords[is_labeled, 0], coords[is_labeled, 1],
               c="#2171b5", s=2.0, alpha=0.40, rasterized=True, zorder=2)

    # Labeled positives as reference
    pos_mask = is_labeled & (bandgaps < threshold)
    n_pos = int(pos_mask.sum())
    if n_pos > 0:
        ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1],
                   c="#2171b5", s=12, marker="^",
                   edgecolors="black", linewidths=0.4,
                   zorder=3, alpha=0.85)

    # Ensemble nominations — big red stars
    n_nom = int(nomination_mask.sum())
    if n_nom > 0:
        ax.scatter(coords[nomination_mask, 0], coords[nomination_mask, 1],
                   c="#e41a1c", s=100, marker="*",
                   edgecolors="black", linewidths=0.5,
                   zorder=5, alpha=0.95)
        # Annotate names
        nom_coords = coords[nomination_mask]
        for k, cid in enumerate(nomination_cids):
            short = cid.replace("_FSR", "")
            if len(short) > 12:
                short = short[:10] + ".."
            ax.annotate(short, (nom_coords[k, 0], nom_coords[k, 1]),
                        fontsize=4.5, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points",
                        color="#333333",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="white", ec="0.7", alpha=0.8, lw=0.3))

    # Build legend with all four categories
    _leg = [
        Line2D([], [], marker="o", color="w", markerfacecolor="#2171b5",
               markersize=5, label=f"DFT-labeled ({n_lab:,})"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#c0c0c0",
               markersize=5, label=f"Unlabeled ({n_unlab:,})"),
    ]
    if n_pos > 0:
        _leg.append(Line2D([], [], marker="^", color="w",
                           markerfacecolor="#2171b5", markeredgecolor="black",
                           markeredgewidth=0.4, markersize=6,
                           label=f"Known positives ({n_pos})"))
    if n_nom > 0:
        _leg.append(Line2D([], [], marker="*", color="w",
                           markerfacecolor="#e41a1c", markeredgecolor="black",
                           markeredgewidth=0.5, markersize=8,
                           label=f"Ensemble nominations ({n_nom})"))
    ax.legend(handles=_leg, loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.95, borderpad=0.5,
              handletextpad=0.5, handlelength=1.4, fontsize=7,
              markerscale=0.8)
    ax.set_title(f"Ensemble nominations  [{exp_label}]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, filename_prefix)
    plt.close()


# ──────────────────────────────────────────────────────────────────────
#  Process one experiment
# ──────────────────────────────────────────────────────────────────────
def process_experiment(npz_path, exp_name, exp_label,
                       labeled_bg, labeled_splits_dict,
                       nomination_cid_list, args):
    """Load NPZ, compute UMAP, generate ensemble nominations panel."""
    print(f"\n{'─' * 60}")
    print(f"  Processing {exp_name}")
    print(f"  NPZ: {npz_path}")
    print(f"{'─' * 60}")

    if not os.path.exists(npz_path):
        print(f"  *** ERROR: NPZ not found: {npz_path}")
        print(f"  *** Skipping {exp_name}")
        return

    # Load embeddings
    cids, embs, bgs_raw, splits_raw, is_lab_raw = load_npz(npz_path)

    # Rebuild metadata from authoritative split labels
    labeled_cid_set = set(labeled_bg.keys())
    if is_lab_raw is not None:
        is_labeled = is_lab_raw
    else:
        is_labeled = np.array([c in labeled_cid_set for c in cids])

    bandgaps = np.array([labeled_bg.get(c, float(bgs_raw[i]))
                         for i, c in enumerate(cids)], dtype=float)

    # Ensemble nomination mask (flexible matching)
    nomination_lookup = {c: True for c in nomination_cid_list}
    nomination_mask = np.array([_flex_lookup(c, nomination_lookup) is not None
                                for c in cids])
    nominations_present = [c for c in cids
                           if _flex_lookup(c, nomination_lookup) is not None]

    n_lab = int(is_labeled.sum())
    n_ulab = len(cids) - n_lab
    n_pos = int((is_labeled & (bandgaps < args.threshold)).sum())
    print(f"    {len(cids)} MOFs ({n_lab} labeled, {n_ulab} unlabeled)")
    print(f"    Known positives (< {args.threshold} eV): {n_pos}")
    print(f"    Ensemble nominations matched: {len(nominations_present)}/{len(nomination_cid_list)}")

    # UMAP — try cache first
    cache_path = os.path.join(args.output_dir, f"umap_cache_{exp_name}.npz")
    coords = None

    if args.load_umap_cache and os.path.exists(cache_path):
        print(f"    Loading cached UMAP coords: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        cached_ids = [str(x) for x in cache["cif_ids"]]
        if cached_ids == cids:
            coords = cache["coords"]
            print(f"    {coords.shape[0]} cached points loaded")
        else:
            print(f"    WARNING: cached IDs differ — recomputing UMAP")

    if coords is None:
        # Also check if the experiment dir has its own umap cache
        exp_dir = os.path.dirname(npz_path)
        exp_umap_cache = os.path.join(exp_dir, "posttrain_umap_cache.npz")
        if os.path.exists(exp_umap_cache):
            print(f"    Found experiment UMAP cache: {exp_umap_cache}")
            cache = np.load(exp_umap_cache, allow_pickle=True)
            cached_ids = [str(x) for x in cache["cif_ids"]]
            if cached_ids == cids:
                coords = cache["coords"]
                print(f"    {coords.shape[0]} cached points loaded from experiment dir")
            else:
                print(f"    WARNING: cached IDs differ — recomputing UMAP")

    if coords is None:
        print(f"    Computing UMAP ...")
        coords = compute_umap(embs, args.n_neighbors, args.min_dist,
                              seed=args.seed)

    # Save UMAP cache
    if args.save_umap_cache:
        np.savez_compressed(cache_path, coords=coords,
                            cif_ids=np.array(cids))
        print(f"    UMAP cache saved: {cache_path}")

    # Filter invalid coords/embeddings
    valid = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    norms = np.linalg.norm(embs, axis=1)
    valid &= norms > 1e-6
    n_skip = int((~valid).sum())
    if n_skip > 0:
        print(f"    Skipping {n_skip} MOFs with invalid embeddings/coords")
        # Rebuild nominations_present from filtered cids
        valid_cids = [c for c, v in zip(cids, valid) if v]
        coords      = coords[valid]
        is_labeled  = is_labeled[valid]
        bandgaps    = bandgaps[valid]
        nomination_mask = nomination_mask[valid]
        nominations_present = [c for c in valid_cids
                          if _flex_lookup(c, nomination_lookup) is not None]
        cids = valid_cids

    # Generate the panel
    filename = f"posttrain_ensemble_{exp_name}"
    panel_ensemble_nominations(
        coords, is_labeled, bandgaps, args.threshold,
        nomination_mask, nominations_present, args.output_dir,
        exp_label, filename)

    # Per-experiment summary
    summary = {
        "experiment": exp_name,
        "total_mofs": len(cids),
        "labeled": int(is_labeled.sum()),
        "unlabeled": len(cids) - int(is_labeled.sum()),
        "known_positives": int((is_labeled & (bandgaps < args.threshold)).sum()),
        "ensemble_nominations_found": len(nominations_present),
        "ensemble_nominations_total": len(nomination_cid_list),
        "embedding_dim": int(embs.shape[1]),
        "umap_params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
            "seed": args.seed,
        },
    }
    sp = os.path.join(args.output_dir, f"summary_{exp_name}.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"    Summary saved: {sp}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(
        description="Post-Training Ensemble Nominations UMAP "
                    "(same as SOAP panel d, but in fine-tuned PMTransformer space)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    pa.add_argument("--npz_exp364", required=True,
                    help="posttrain_embeddings.npz for exp364")
    pa.add_argument("--npz_exp370", required=True,
                    help="posttrain_embeddings.npz for exp370")
    pa.add_argument("--npz_exp371", required=True,
                    help="posttrain_embeddings.npz for exp371")
    pa.add_argument("--labeled_splits_dir", required=True,
                    help="Dir with {train,val,test}_bandgaps_regression.json")
    pa.add_argument("--nominations", required=True,
                    help="File with top-25 ensemble CIF IDs (one per line)")
    pa.add_argument("--output_dir", default="./figures_output/ensemble_umap")
    pa.add_argument("--threshold", type=float, default=1.0,
                    help="Bandgap threshold for positive class (eV)")
    pa.add_argument("--n_neighbors", type=int, default=30)
    pa.add_argument("--min_dist", type=float, default=0.3)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--load_umap_cache", action="store_true",
                    help="Load cached UMAP coords if available")
    pa.add_argument("--save_umap_cache", action="store_true",
                    help="Save UMAP coords for fast re-runs")

    args = pa.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_publication_style()

    print("=" * 70)
    print("  POST-TRAINING ENSEMBLE NOMINATIONS UMAP")
    print("  (same as SOAP panel d, in fine-tuned PMTransformer space)")
    print("=" * 70)

    # ── Load shared metadata ──────────────────────────────────────────
    print("\n[1] Loading labels ...")
    labeled_bg, labeled_splits_dict = load_split_labels(
        os.path.abspath(args.labeled_splits_dir))
    print(f"    Total labeled: {len(labeled_bg)}")

    print("\n[2] Loading ensemble nominations ...")
    nomination_cid_list = load_nominations(
        os.path.abspath(args.nominations))

    # ── Process each experiment ───────────────────────────────────────
    experiments = [
        (args.npz_exp364, "exp364", "fine-tuned exp364"),
        (args.npz_exp370, "exp370", "fine-tuned exp370"),
        (args.npz_exp371, "exp371", "fine-tuned exp371"),
    ]

    print(f"\n[3] Generating plots for {len(experiments)} experiments ...")
    for npz_path, exp_name, exp_label in experiments:
        process_experiment(
            os.path.abspath(npz_path), exp_name, exp_label,
            labeled_bg, labeled_splits_dict,
            nomination_cid_list, args)

    print(f"\n{'=' * 70}")
    print(f"  Done.  All outputs in {args.output_dir}/")
    print(f"  Files: posttrain_ensemble_exp364/370/371 (.png/.svg/.pdf)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
