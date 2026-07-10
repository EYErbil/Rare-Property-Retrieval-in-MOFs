#!/usr/bin/env python3
"""
Figure 2: Publication-Quality UMAP Chemical Space Visualization
================================================================

Four separate panels saved as individual PNG + SVG:

  (a) All MOFs: DFT-labeled vs. unlabeled
  (b) Labeled MOFs: colored by DFT bandgap (viridis, 0–5 eV)
  (c) All MOFs: colored by primary metal center (from qmof.csv)
  (d) Labeled-only zoomed view:  train / val / test split assignments

Recommended workflow
--------------------
1. Run forward_pretrained_embeddings.py ONCE to get aligned embeddings
   for all ~20K MOFs in a single forward pass.
2. Run this script with --merged_embeddings pointing to that NPZ.

Usage
-----
  # From merged embeddings (RECOMMENDED — produced by forward_pretrained_embeddings.py):
  python figures/umap_pretrained.py \\
      --merged_embeddings figures_output/pretrained_embeddings/all_embeddings.npz \\
      --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --qmof_csv data/qmof.csv \\
      --output_dir figures_output/pretrained_umap

  # From separate embeddings (legacy, alignment risk):
  python figures/umap_pretrained.py \\
      --labeled_embeddings data/embeddings/embeddings_pretrained.npz \\
      --unlabeled_embeddings data/unlabeled/embedding_analysis/unlabeled_embeddings.npz \\
      --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --qmof_csv data/qmof.csv \\
      --output_dir figures_output/pretrained_umap

Requirements:  pip install numpy matplotlib umap-learn
"""

import os
import sys
import json
import re
import csv as csvmod
import argparse
import numpy as np
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────
ORGANIC_ELEMENTS = frozenset({
    "C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I",
    "Si", "B", "Se", "Te", "As", "Ge", "Sb",
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
})

# ──────────────────────────────────────────────────────────────────────
#  Publication style
# ──────────────────────────────────────────────────────────────────────
def set_publication_style():
    """Configure matplotlib for Nature / ACS requirements."""
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
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "lines.linewidth":   0.5,
        "patch.linewidth":   0.4,
        "pdf.fonttype":      42,   # TrueType – required by most journals
        "ps.fonttype":       42,
        "mathtext.default":  "regular",
    })


# ──────────────────────────────────────────────────────────────────────
#  Data loading helpers
# ──────────────────────────────────────────────────────────────────────
def load_npz_embeddings(npz_path):
    """Load {cif_ids, embeddings, bandgaps, splits} from an npz file."""
    data = np.load(npz_path, allow_pickle=True)
    raw_ids    = data["cif_ids"]
    # Ensure plain Python str (not numpy.str_, not bytes)
    cif_ids    = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                  for x in raw_ids]
    embeddings = data["embeddings"]
    bandgaps   = data["bandgaps"] if "bandgaps" in data else np.full(len(cif_ids), np.nan)
    splits     = list(data["splits"]) if "splits" in data else ["unknown"] * len(cif_ids)
    is_labeled = data["is_labeled"] if "is_labeled" in data else None
    # Debug: show type + sample
    print(f"    NPZ raw type: {type(raw_ids[0])}  ->  converted: {type(cif_ids[0])}")
    print(f"    First 3 IDs: {cif_ids[:3]}")
    return cif_ids, embeddings, bandgaps, splits, is_labeled


def load_split_labels(splits_dir):
    """Return (cif_id->bandgap, cif_id->split) dicts from split JSONs."""
    if not os.path.isdir(splits_dir):
        print(f"  *** WARNING: labeled_splits_dir does not exist: {splits_dir}")
        print(f"  *** Check your --labeled_splits_dir path!")
        return {}, {}
    labels, assignments = {}, {}
    found_any = False
    for split_name in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{split_name}_bandgaps_regression.json")
        if not os.path.exists(p):
            print(f"    {split_name}: NOT FOUND at {p}")
            continue
        found_any = True
        with open(p) as fh:
            d = json.load(fh)
        for cid, bg in d.items():
            labels[cid] = float(bg)
            assignments[cid] = split_name
        print(f"    {split_name}: {len(d)} MOFs loaded")
    if not found_any:
        print(f"  *** WARNING: No split JSONs found in {splits_dir}")
        print(f"  *** Expected files: train_bandgaps_regression.json, etc.")
        print(f"  *** Directory contents: {os.listdir(splits_dir)[:20]}")
    return labels, assignments


def load_unlabeled_ids(json_path):
    """Load unlabeled CIF IDs from inference JSON (values are dummy 0)."""
    with open(json_path) as fh:
        return set(json.load(fh).keys())


# ──────────────────────────────────────────────────────────────────────
#  Metal-center extraction  (qmof.csv — authoritative source)
# ──────────────────────────────────────────────────────────────────────
def _metals_from_formula(formula):
    """Return list of (element, count) pairs for metal elements."""
    tokens = re.findall(r"([A-Z][a-z]?)\s*(\d*\.?\d*)", formula)
    metals = []
    for elem, cnt in tokens:
        if elem not in ORGANIC_ELEMENTS:
            metals.append((elem, float(cnt) if cnt else 1.0))
    return metals


def get_metals_from_qmof_csv(cif_ids, csv_path):
    """Return {cif_id: metal} from qmof.csv (the ground-truth source).

    qmof.csv columns used:
      - ``name``           CIF ID (matched with flexible normalization)
      - ``qmof_id``        alternate identifier
      - ``info.formula``   chemical formula string

    Matching strategy (tried in order for each CIF ID):
      1. Exact match on ``name``
      2. After stripping ``.cif`` extension
      3. After stripping ``_FSR`` suffix
      4. After stripping both ``.cif`` and ``_FSR``
      5. Match on ``qmof_id``
    """
    # Build lookup tables from qmof.csv
    name_to_formula = {}       # name -> formula
    qmofid_to_formula = {}     # qmof_id -> formula
    stripped_to_formula = {}    # name without _FSR -> formula (first wins)

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csvmod.DictReader(fh)
        for row in reader:
            name = row.get("name", "").strip()
            qmof_id = row.get("qmof_id", "").strip()
            formula = row.get("info.formula", "").strip()
            if not formula:
                continue
            if name:
                name_to_formula[name] = formula
                # Also index without _FSR suffix
                bare = name[:-4] if name.endswith("_FSR") else name
                if bare not in stripped_to_formula:
                    stripped_to_formula[bare] = formula
            if qmof_id:
                qmofid_to_formula[qmof_id] = formula
    print(f"    qmof.csv: {len(name_to_formula)} entries loaded "
          f"(+ {len(qmofid_to_formula)} qmof_ids, "
          f"{len(stripped_to_formula)} bare names)")

    def _lookup(cid):
        """Try multiple matching strategies."""
        # 1. Exact match
        if cid in name_to_formula:
            return name_to_formula[cid]
        # 2. Strip .cif extension
        bare_cid = cid[:-4] if cid.lower().endswith(".cif") else cid
        if bare_cid != cid and bare_cid in name_to_formula:
            return name_to_formula[bare_cid]
        # 3. Strip _FSR suffix
        no_fsr = bare_cid[:-4] if bare_cid.endswith("_FSR") else bare_cid
        if no_fsr in stripped_to_formula:
            return stripped_to_formula[no_fsr]
        # 4. Try with _FSR added
        with_fsr = bare_cid + "_FSR"
        if with_fsr in name_to_formula:
            return name_to_formula[with_fsr]
        # 5. qmof_id match
        if bare_cid in qmofid_to_formula:
            return qmofid_to_formula[bare_cid]
        return ""

    # Show diagnostic: first 5 CIF IDs vs first 5 qmof.csv names
    sample_cids = list(cif_ids[:5]) if len(cif_ids) >= 5 else list(cif_ids)
    sample_names = list(name_to_formula.keys())[:5]
    print(f"    Sample CIF IDs (type={type(sample_cids[0]) if sample_cids else '?'}):")
    for sc in sample_cids:
        print(f"      {repr(sc)}")
    print(f"    Sample qmof names:")
    for sn in sample_names:
        print(f"      {repr(sn)}")
    # Test first CIF ID against all strategies
    if sample_cids:
        test_cid = sample_cids[0]
        print(f"    Debug lookup for {repr(test_cid)}:")
        print(f"      exact match in name_to_formula: {test_cid in name_to_formula}")
        bare = test_cid[:-4] if test_cid.lower().endswith('.cif') else test_cid
        print(f"      after .cif strip: {repr(bare)} -> in name_to_formula: {bare in name_to_formula}")
        no_fsr = bare[:-4] if bare.endswith('_FSR') else bare
        print(f"      after _FSR strip: {repr(no_fsr)} -> in stripped_to_formula: {no_fsr in stripped_to_formula}")
        with_fsr = bare + '_FSR'
        print(f"      add _FSR: {repr(with_fsr)} -> in name_to_formula: {with_fsr in name_to_formula}")
        print(f"      in qmofid_to_formula: {bare in qmofid_to_formula}")
        result = _lookup(test_cid)
        print(f"      => _lookup result: {repr(result)}")

    metals = {}
    n_miss, n_no_metal = 0, 0
    for cid in cif_ids:
        formula = _lookup(cid)
        if not formula:
            metals[cid] = "Unknown"
            n_miss += 1
            continue
        m = _metals_from_formula(formula)
        if m:
            # pick the metal with the highest stoichiometric count
            m.sort(key=lambda x: -x[1])
            metals[cid] = m[0][0]
        else:
            metals[cid] = "Unknown"
            n_no_metal += 1
    if n_miss:
        print(f"    WARNING: {n_miss}/{len(cif_ids)} MOFs not found in qmof.csv")
    if n_no_metal:
        print(f"    WARNING: {n_no_metal} MOFs had formulas with no detected metal")
    n_found = len(cif_ids) - n_miss - n_no_metal
    print(f"    Metal extraction: {n_found} metals found, "
          f"{n_miss} not matched, {n_no_metal} no metal in formula")
    return metals


# ──────────────────────────────────────────────────────────────────────
#  UMAP
# ──────────────────────────────────────────────────────────────────────
def compute_umap(embeddings, n_neighbors=30, min_dist=0.3, metric="cosine",
                 seed=42):
    try:
        from umap import UMAP
    except ImportError:
        sys.exit("ERROR: umap-learn not installed.  pip install umap-learn")

    print(f"    {embeddings.shape[0]} points, dim={embeddings.shape[1]} ...")
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                   metric=metric, random_state=seed, n_jobs=-1)
    coords = reducer.fit_transform(embeddings)
    print(f"    UMAP done -> shape {coords.shape}")
    return coords


# ──────────────────────────────────────────────────────────────────────
#  Axis helper  (no axis labels — UMAP dims are arbitrary)
# ──────────────────────────────────────────────────────────────────────
def _style_ax(ax, space="NN embedding"):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=7, labelpad=2)
    ax.set_ylabel("UMAP-2", fontsize=7, labelpad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color("0.5")


def _save_panel(fig, output_dir, name):
    """Save a single-panel figure as PNG, SVG, and PDF."""
    for fmt in ("png", "svg", "pdf"):
        p = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(p, dpi=600 if fmt == "png" else 300,
                    format=fmt, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    print(f"    Saved {name}.png / .svg / .pdf")


# ──────────────────────────────────────────────────────────────────────
#  Panel (a): Labeled vs Unlabeled
# ──────────────────────────────────────────────────────────────────────
def panel_a_labeled_unlabeled(coords, is_labeled, output_dir,
                              figsize=(4.5, 4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    unlabeled_mask = ~is_labeled
    n_lab   = int(is_labeled.sum())
    n_unlab = int(unlabeled_mask.sum())

    ax.scatter(coords[unlabeled_mask, 0], coords[unlabeled_mask, 1],
               c="#a0a0a0", s=2.0, alpha=0.40, rasterized=True, zorder=1)
    ax.scatter(coords[is_labeled, 0], coords[is_labeled, 1],
               c="#2171b5", s=2.5, alpha=0.55, rasterized=True, zorder=2)

    _leg = [
        Line2D([], [], marker="o", color="w", markerfacecolor="#2171b5",
               markersize=5, label=f"DFT-labeled ({n_lab:,})"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#a0a0a0",
               markersize=5, label=f"Unlabeled ({n_unlab:,})"),
    ]
    ax.legend(handles=_leg, loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.95, borderpad=0.5,
              handletextpad=0.5, handlelength=1.4, fontsize=7)
    ax.set_title("(a)  Labeled vs. unlabeled", fontweight="bold", pad=6,
                 fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "fig2a_labeled_vs_unlabeled")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
#  Panel (b): DFT bandgap — discrete bins
# ──────────────────────────────────────────────────────────────────────
BANDGAP_BINS = [
    (0.0, 0.5,  "#d73027",  "0 – 0.5 eV"),    # deep red
    (0.5, 1.0,  "#fc8d59",  "0.5 – 1 eV"),     # orange
    (1.0, 2.0,  "#fee08b",  "1 – 2 eV"),       # yellow
    (2.0, 3.0,  "#d9ef8b",  "2 – 3 eV"),       # light green
    (3.0, 4.0,  "#91bfdb",  "3 – 4 eV"),       # light blue
    (4.0, 99.0, "#4575b4",  "≥ 4 eV"),         # deep blue
]


def panel_b_bandgap(coords, is_labeled, bandgaps, threshold, output_dir,
                    figsize=(4.8, 4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    unlabeled_mask = ~is_labeled

    # Background: unlabeled MOFs
    ax.scatter(coords[unlabeled_mask, 0], coords[unlabeled_mask, 1],
               c="#c0c0c0", s=1.5, alpha=0.30, rasterized=True, zorder=1)

    # Plot labeled MOFs in discrete bandgap bins (widest gap first → narrowest on top)
    lab_coords = coords[is_labeled]
    lab_bg = bandgaps[is_labeled]
    legend_handles = []

    for lo, hi, color, label in reversed(BANDGAP_BINS):
        mask = (lab_bg >= lo) & (lab_bg < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        ax.scatter(lab_coords[mask, 0], lab_coords[mask, 1],
                   c=color, s=2.5, alpha=0.65, rasterized=True, zorder=2)
        legend_handles.append(
            Line2D([], [], marker="o", color="w", markerfacecolor=color,
                   markersize=5, label=f"{label}  ({n:,})"))

    # Re-order legend entries from low → high bandgap
    legend_handles.reverse()

    # Unlabeled entry in legend
    n_ulab = int(unlabeled_mask.sum())
    legend_handles.append(
        Line2D([], [], marker="o", color="w", markerfacecolor="#c0c0c0",
               markersize=5, label=f"Unlabeled ({n_ulab:,})"))

    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              fancybox=False, edgecolor="0.7", framealpha=0.95,
              borderpad=0.5, handletextpad=0.5, handlelength=1.4,
              fontsize=6, ncol=1)
    ax.set_title("(b)  DFT bandgap", fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "fig2b_bandgap")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
#  Panel (c): Primary metal center
# ──────────────────────────────────────────────────────────────────────
def panel_c_metal(coords, metals, n_top_metals, output_dir,
                  figsize=(4.8, 4.0)):
    fig, ax = plt.subplots(figsize=figsize)

    metal_counts = Counter(metals)
    top_metals = [m for m, _ in metal_counts.most_common()
                  if m != "Unknown"][:n_top_metals]

    # Use distinct, colorblind-friendly palette
    _palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                "#ff7f00", "#a65628", "#f781bf", "#999999",
                "#66c2a5", "#fc8d62"]
    metal_colors = {m: _palette[i % len(_palette)]
                    for i, m in enumerate(top_metals)}
    metal_colors["Other"] = "#d9d9d9"

    metal_arr = np.array(metals)
    label_arr = np.where(np.isin(metal_arr, top_metals), metal_arr, "Other")

    other_mask = label_arr == "Other"
    ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
               c=metal_colors["Other"], s=1.5, alpha=0.30,
               rasterized=True, zorder=1)

    for m in top_metals:
        mask = label_arr == m
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[metal_colors[m]], s=2.5, alpha=0.50,
                   rasterized=True, zorder=2)

    _leg = [Line2D([], [], marker="o", color="w",
                   markerfacecolor=metal_colors[m], markersize=5,
                   label=f"{m} ({(label_arr == m).sum():,})")
            for m in top_metals if (label_arr == m).sum() > 0]
    _leg.append(Line2D([], [], marker="o", color="w",
                       markerfacecolor=metal_colors["Other"], markersize=5,
                       label=f"Other ({int(other_mask.sum()):,})"))
    ncol = 2 if len(top_metals) > 5 else 1
    ax.legend(handles=_leg, loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.95, borderpad=0.5,
              columnspacing=0.6, handletextpad=0.5, handlelength=1.4,
              ncol=ncol, fontsize=6.5)
    ax.set_title("(c)  Primary metal center", fontweight="bold", pad=6,
                 fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "fig2c_metal_center")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
#  Panel (d): Labeled-only, zoomed, train / val / test
# ──────────────────────────────────────────────────────────────────────
def panel_d_labeled_zoom(coords, is_labeled, bandgaps, split_labels,
                         threshold, output_dir, figsize=(4.8, 4.0)):
    """Zoomed view of only DFT-labeled MOFs with train/val/test coloring."""
    fig, ax = plt.subplots(figsize=figsize)

    lab_coords = coords[is_labeled]
    lab_bgs    = bandgaps[is_labeled]
    lab_splits = split_labels[is_labeled]

    pos_mask = lab_bgs < threshold

    colors_map = {"train": "#4292c6", "val": "#fd8d3c", "test": "#969696"}
    order  = ["test", "val", "train"]   # paint train on top

    for sp in order:
        mask = lab_splits == sp
        n = int(mask.sum())
        ax.scatter(lab_coords[mask, 0], lab_coords[mask, 1],
                   c=colors_map.get(sp, "#cccccc"), s=2.5, alpha=0.40,
                   rasterized=True, zorder=1 if sp == "test" else 2,
                   label=f"{sp.capitalize()} ({n:,})")

    # highlight positives per split
    for sp, marker, sz, zorder, color in [
        ("train", "^", 45, 4, "#08519c"),
        ("val",   "D", 35, 4, "#d94801"),
        ("test",  "*", 80, 5, "#e41a1c"),
    ]:
        mask = (lab_splits == sp) & pos_mask
        n = int(mask.sum())
        if n == 0:
            continue
        ax.scatter(lab_coords[mask, 0], lab_coords[mask, 1],
                   c=color, s=sz, marker=marker,
                   edgecolors="black", linewidths=0.5,
                   zorder=zorder, alpha=0.90,
                   label=f"{sp.capitalize()} positive ({n})")

    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.95, borderpad=0.5,
              handletextpad=0.5, handlelength=1.4, fontsize=6.5,
              markerscale=1.3)
    ax.set_title("(d)  Labeled MOFs — train / val / test",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "fig2d_labeled_splits_zoom")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(
        description="Figure 2: UMAP Chemical Space (Publication Quality)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # -- embeddings --
    g = pa.add_argument_group("Embedding inputs (EITHER merged OR labeled+unlabeled)")
    g.add_argument("--labeled_embeddings", type=str, default=None,
                   help="embeddings_pretrained.npz for labeled MOFs")
    g.add_argument("--unlabeled_embeddings", type=str, default=None,
                   help="Embeddings .npz for unlabeled MOFs")
    g.add_argument("--merged_embeddings", type=str, default=None,
                   help="Unified NPZ from forward_pretrained_embeddings.py "
                        "(RECOMMENDED)")

    # -- labels --
    g2 = pa.add_argument_group("Label inputs")
    g2.add_argument("--labeled_splits_dir", type=str, required=True,
                    help="Dir with {train,val,test}_bandgaps_regression.json")
    g2.add_argument("--unlabeled_json", type=str, default=None,
                    help="Unlabeled test_bandgaps_regression.json (optional)")

    # -- metal source --
    g3 = pa.add_argument_group("Metal-center source")
    g3.add_argument("--qmof_csv", type=str, default=None,
                    help="qmof.csv (columns: name, info.formula). "
                         "RECOMMENDED — authoritative source.")

    # -- output & UMAP --
    pa.add_argument("--output_dir", type=str, default="./figures_output/pretrained_umap")
    pa.add_argument("--threshold", type=float, default=1.0,
                    help="Bandgap threshold (eV) for 'positive' class")
    pa.add_argument("--n_top_metals", type=int, default=8)
    pa.add_argument("--n_neighbors", type=int, default=30)
    pa.add_argument("--min_dist", type=float, default=0.3)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--load_cache", type=str, default=None,
                    help="NPZ with cached UMAP coords (skip recomputation)")
    pa.add_argument("--save_cache", action="store_true",
                    help="Write UMAP coords to output_dir/umap_cache.npz")

    args = pa.parse_args()

    set_publication_style()

    print("=" * 70)
    print("  FIGURE 2 — UMAP Chemical Space Visualization")
    print("=" * 70)

    # ── 1. Labelled bandgap info ──────────────────────────────────────
    print("\n[1/5] Loading labeled split information ...")
    labeled_bg, labeled_splits_dict = load_split_labels(args.labeled_splits_dir)
    labeled_cid_set = set(labeled_bg.keys())
    print(f"       Total labeled MOFs: {len(labeled_cid_set)}")

    unlabeled_cid_set = set()
    if args.unlabeled_json and os.path.exists(args.unlabeled_json):
        unlabeled_cid_set = load_unlabeled_ids(args.unlabeled_json)
        print(f"       Total unlabeled MOFs from JSON: {len(unlabeled_cid_set)}")

    # ── 2. Load embeddings ────────────────────────────────────────────
    print("\n[2/5] Loading embeddings ...")
    if args.merged_embeddings:
        print(f"       Unified file: {args.merged_embeddings}")
        cids, embs, bgs_raw, splits_raw, is_lab_raw = load_npz_embeddings(
            args.merged_embeddings)
        # Determine labeled status
        if is_lab_raw is not None:
            is_labeled = np.asarray(is_lab_raw, dtype=bool)
        else:
            is_labeled = np.array([c in labeled_cid_set for c in cids])
        # Override bandgaps with authoritative values where available
        bgs = np.array([labeled_bg.get(c, float(bgs_raw[i]))
                        for i, c in enumerate(cids)], dtype=float)
        # Build split label array
        split_labels = np.array([labeled_splits_dict.get(c, "unlabeled")
                                 for c in cids])

    elif args.labeled_embeddings and args.unlabeled_embeddings:
        print(f"       Labeled: {args.labeled_embeddings}")
        c5, e5, b5, _, _ = load_npz_embeddings(args.labeled_embeddings)
        print(f"         -> {len(c5)} MOFs, dim {e5.shape[1]}")

        print(f"       Unlabeled: {args.unlabeled_embeddings}")
        c6, e6, b6, _, _ = load_npz_embeddings(args.unlabeled_embeddings)
        print(f"         -> {len(c6)} MOFs, dim {e6.shape[1]}")

        seen = set()
        m_cids, m_embs, m_lab, m_bgs, m_spl = [], [], [], [], []
        for i, cid in enumerate(c5):
            if cid in seen:
                continue
            seen.add(cid)
            m_cids.append(cid)
            m_embs.append(e5[i])
            m_lab.append(cid in labeled_cid_set)
            m_bgs.append(labeled_bg.get(cid, float(b5[i])))
            m_spl.append(labeled_splits_dict.get(cid, "unlabeled"))
        for i, cid in enumerate(c6):
            if cid in seen:
                continue
            seen.add(cid)
            m_cids.append(cid)
            m_embs.append(e6[i])
            m_lab.append(cid in labeled_cid_set)
            m_bgs.append(labeled_bg.get(cid, np.nan))
            m_spl.append(labeled_splits_dict.get(cid, "unlabeled"))

        cids         = m_cids
        embs         = np.array(m_embs)
        is_labeled   = np.array(m_lab)
        bgs          = np.array(m_bgs, dtype=float)
        split_labels = np.array(m_spl)
        print(f"       Merged: {len(cids)} unique  "
              f"({is_labeled.sum()} labeled, {(~is_labeled).sum()} unlabeled)")
        print("       *** WARNING: separate forward passes — embeddings may "
              "not be perfectly aligned. Consider using "
              "forward_pretrained_embeddings.py instead. ***")
    else:
        sys.exit("ERROR: provide --merged_embeddings  OR  "
                 "--labeled_embeddings + --unlabeled_embeddings")

    # ── 3. UMAP ──────────────────────────────────────────────────────
    if args.load_cache and os.path.exists(args.load_cache):
        print(f"\n[3/5] Loading cached UMAP coords: {args.load_cache}")
        cache = np.load(args.load_cache, allow_pickle=True)
        coords = cache["coords"]
        cached_ids = list(cache["cif_ids"])
        if cached_ids != list(cids):
            print("       WARNING: cached IDs differ — recomputing UMAP")
            coords = compute_umap(embs, args.n_neighbors, args.min_dist,
                                  seed=args.seed)
        else:
            print(f"       {coords.shape[0]} cached points loaded")
    else:
        print(f"\n[3/5] Computing UMAP projection ...")
        coords = compute_umap(embs, args.n_neighbors, args.min_dist,
                              seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_cache:
        cp = os.path.join(args.output_dir, "umap_cache.npz")
        np.savez_compressed(cp, coords=coords, cif_ids=np.array(cids))
        print(f"       Cache saved: {cp}")

    # ── 4. Metal centers from qmof.csv ────────────────────────────────
    print(f"\n[4/5] Extracting metal centers ...")
    if args.qmof_csv and os.path.exists(args.qmof_csv):
        metals_dict = get_metals_from_qmof_csv(cids, args.qmof_csv)
    else:
        print("       No --qmof_csv provided — panel (c) will use 'Unknown'")
        metals_dict = {c: "Unknown" for c in cids}
    metals = np.array([metals_dict.get(c, "Unknown") for c in cids])

    # ── 4b. Drop invalid MOFs ─────────────────────────────────────────
    valid = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    norms = np.linalg.norm(embs, axis=1)
    valid &= norms > 1e-6

    n_skip = int((~valid).sum())
    if n_skip > 0:
        print(f"       Skipping {n_skip} MOFs with invalid embeddings/coords")
        coords       = coords[valid]
        is_labeled   = is_labeled[valid]
        bgs          = bgs[valid]
        metals       = metals[valid]
        split_labels = split_labels[valid]
        cids         = [c for c, v in zip(cids, valid) if v]
        embs         = embs[valid]

    print(f"       Final: {len(cids)} MOFs  "
          f"({is_labeled.sum()} labeled, {(~is_labeled).sum()} unlabeled)")

    # ── 4c. Matching verification ─────────────────────────────────────
    print(f"\n── Matching verification ──")
    n_total = len(cids)
    n_lab   = int(is_labeled.sum())
    n_ulab  = n_total - n_lab

    # Check bandgap metadata: labeled must have finite bandgap, unlabeled NaN
    lab_bgs  = bgs[is_labeled]
    ulab_bgs = bgs[~is_labeled]
    n_lab_finite = int(np.isfinite(lab_bgs).sum())
    n_ulab_nan   = int(np.isnan(ulab_bgs).sum())
    print(f"  Labeled  : {n_lab} total, {n_lab_finite} have finite bandgap")
    if n_lab_finite < n_lab:
        print(f"    WARNING: {n_lab - n_lab_finite} labeled MOFs have NaN bandgap!")
    print(f"  Unlabeled: {n_ulab} total, {n_ulab_nan} have NaN bandgap (expected)")
    n_ulab_finite = int(np.isfinite(ulab_bgs).sum())
    if n_ulab_finite > 0:
        print(f"    WARNING: {n_ulab_finite} unlabeled MOFs have finite bandgap "
              f"— these may be mis-classified")

    # Check split labels
    from collections import Counter as _C
    split_counts = _C(split_labels)
    print(f"  Split distribution: ", end="")
    for sp in ("train", "val", "test", "unlabeled"):
        print(f"{sp}={split_counts.get(sp, 0):,}  ", end="")
    other_splits = {k: v for k, v in split_counts.items()
                    if k not in ("train", "val", "test", "unlabeled")}
    if other_splits:
        print(f"OTHER={other_splits}", end="")
    print()

    # Cross-check: split labeled must match is_labeled
    split_lab_mask = np.isin(split_labels, ["train", "val", "test"])
    mismatch = int((split_lab_mask != is_labeled).sum())
    if mismatch > 0:
        print(f"  WARNING: {mismatch} MOFs have split/is_labeled mismatch!")
    else:
        print(f"  Split/is_labeled cross-check: OK")

    # Check metal coverage
    n_known_metal = int((metals != "Unknown").sum())
    print(f"  Metal centers: {n_known_metal}/{n_total} identified "
          f"({n_known_metal/n_total*100:.1f}%)")

    # Bandgap range (labeled only, for sanity)
    if n_lab_finite > 0:
        print(f"  Bandgap range (labeled): "
              f"{np.nanmin(lab_bgs):.3f} – {np.nanmax(lab_bgs):.3f} eV  "
              f"(mean {np.nanmean(lab_bgs):.3f}, std {np.nanstd(lab_bgs):.3f})")
        n_pos = int((lab_bgs < args.threshold).sum())
        print(f"  Low bandgap (<{args.threshold} eV): {n_pos} MOFs "
              f"({n_pos/n_lab*100:.1f}% of labeled)")

    print(f"── End verification ──\n")

    # ── 5. Generate panels ────────────────────────────────────────────
    print(f"[5/5] Generating panels ...")

    panel_a_labeled_unlabeled(coords, is_labeled, args.output_dir)
    panel_b_bandgap(coords, is_labeled, bgs, args.threshold, args.output_dir)
    panel_c_metal(coords, metals, args.n_top_metals, args.output_dir)
    panel_d_labeled_zoom(coords, is_labeled, bgs, split_labels,
                         args.threshold, args.output_dir)

    # ── Summary JSON ──────────────────────────────────────────────────
    mc = Counter(metals)
    stats = {
        "total_mofs":      len(cids),
        "labeled":         int(is_labeled.sum()),
        "unlabeled":       int((~is_labeled).sum()),
        "n_low_bandgap":   int(np.nansum(bgs[is_labeled] < args.threshold)),
        "bandgap_stats": {
            "mean":   float(np.nanmean(bgs[is_labeled])),
            "median": float(np.nanmedian(bgs[is_labeled])),
            "std":    float(np.nanstd(bgs[is_labeled])),
            "min":    float(np.nanmin(bgs[is_labeled])),
            "max":    float(np.nanmax(bgs[is_labeled])),
        },
        "top_metals":  dict(mc.most_common(15)),
        "skipped":     n_skip,
    }
    sp = os.path.join(args.output_dir, "figure2_stats.json")
    with open(sp, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"    Stats -> {sp}")

    print(f"\n{'='*70}")
    print(f"  Done.  4 panels saved in {args.output_dir}/")
    print(f"    fig2a_labeled_vs_unlabeled.png / .svg / .pdf")
    print(f"    fig2b_bandgap.png / .svg / .pdf")
    print(f"    fig2c_metal_center.png / .svg / .pdf")
    print(f"    fig2d_labeled_splits_zoom.png / .svg / .pdf")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
