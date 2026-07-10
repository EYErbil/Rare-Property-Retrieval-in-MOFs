#!/usr/bin/env python3
"""
Nominated Structures UMAP with DFT Bandgap — Pretrained PMTransformer Space
============================================================================

Plots all MOFs in 2D UMAP (from pretrained PMTransformer forward-pass
embeddings), and highlights the 25 nominated structures color-coded by
their DFT bandgap from bandgap_results.csv.

Two panels:
  (a) Overview — all MOFs in gray/blue, nominations as colored stars (by bandgap)
  (b) Zoom on nominations — annotated with MOF name + bandgap value

Usage (cluster):
  # Pretrained (foundational) model embeddings:
  python figure_nominated_bandgap_umap.py \
      --pretrained_npz /path/to/project/embeddings/pmt_embeddings_qmof_all.npz \
      --bandgap_csv /path/to/project/nominated/bandgap_results.csv \
      --labeled_splits_dir /path/to/new_splits/strategy_d_farthest_point \
      --output_dir ./nominated_umap_figures

  # Also overlay a fine-tuned model for comparison:
  python figure_nominated_bandgap_umap.py \
      --pretrained_npz /path/to/embeddings/pmt_embeddings_qmof_all.npz \
      --finetuned_npz /path/to/posttrain_umap_figures_exp370/posttrain_embeddings.npz \
      --finetuned_name exp370 \
      --bandgap_csv /path/to/bandgap_results.csv \
      --labeled_splits_dir /path/to/new_splits/strategy_d_farthest_point \
      --output_dir ./nominated_umap_figures

  # With UMAP cache for fast re-runs:
  python figure_nominated_bandgap_umap.py \
      --pretrained_npz /path/to/all_embeddings.npz \
      --bandgap_csv /path/to/bandgap_results.csv \
      --labeled_splits_dir /path/to/new_splits/strategy_d_farthest_point \
      --output_dir ./nominated_umap_figures \
      --load_umap_cache --save_umap_cache

Requirements: pip install numpy matplotlib umap-learn
"""

import os
import sys
import json
import csv
import re
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ──────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────
BANDGAP_THRESHOLD = 1.0  # eV — positive class boundary
ORGANIC_ELEMENTS = frozenset({
    "C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I",
    "Si", "B", "Se", "Te", "As", "Ge", "Sb",
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
})


# ──────────────────────────────────────────────────────────────────────
#  Publication style (consistent with other UMAP scripts)
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
#  Flexible ID matching (reused from other scripts)
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


def _metals_from_formula(formula):
    """Return list of (element, count) pairs for metal elements."""
    tokens = re.findall(r"([A-Z][a-z]?)\s*(\d*\.?\d*)", formula)
    metals = []
    for elem, cnt in tokens:
        if elem not in ORGANIC_ELEMENTS:
            metals.append((elem, float(cnt) if cnt else 1.0))
    return metals


def get_metals_from_qmof_csv(cif_ids, csv_path):
    """Return {cif_id: metal} from qmof.csv using flexible ID matching."""
    name_to_formula = {}
    qmofid_to_formula = {}
    stripped_to_formula = {}

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("name", "").strip()
            qmof_id = row.get("qmof_id", "").strip()
            formula = row.get("info.formula", "").strip()
            if not formula:
                continue
            if name:
                name_to_formula[name] = formula
                bare = name[:-4] if name.endswith("_FSR") else name
                if bare not in stripped_to_formula:
                    stripped_to_formula[bare] = formula
            if qmof_id:
                qmofid_to_formula[qmof_id] = formula

    def _lookup_formula(cid):
        if cid in name_to_formula:
            return name_to_formula[cid]
        bare_cid = cid[:-4] if cid.lower().endswith(".cif") else cid
        if bare_cid != cid and bare_cid in name_to_formula:
            return name_to_formula[bare_cid]
        no_fsr = bare_cid[:-4] if bare_cid.endswith("_FSR") else bare_cid
        if no_fsr in stripped_to_formula:
            return stripped_to_formula[no_fsr]
        with_fsr = bare_cid + "_FSR"
        if with_fsr in name_to_formula:
            return name_to_formula[with_fsr]
        if bare_cid in qmofid_to_formula:
            return qmofid_to_formula[bare_cid]
        return ""

    metals = {}
    for cid in cif_ids:
        formula = _lookup_formula(cid)
        if not formula:
            metals[cid] = "Unknown"
            continue
        m = _metals_from_formula(formula)
        if not m:
            metals[cid] = "Unknown"
            continue
        m.sort(key=lambda x: -x[1])
        metals[cid] = m[0][0]
    return metals


# ──────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────
def load_npz(npz_path):
    """Load posttrain_embeddings.npz."""
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
    """Load bandgap labels from split JSONs."""
    if not os.path.isdir(splits_dir):
        print(f"  *** WARNING: labeled_splits_dir does not exist: {splits_dir}")
        return {}
    labels = {}
    for split_name in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{split_name}_bandgaps_regression.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        for cid, bg in d.items():
            labels[cid] = float(bg)
        print(f"    {split_name}: {len(d)} MOFs")
    return labels


def load_bandgap_csv(csv_path):
    """Load DFT bandgap results CSV.

    Returns dict: {folder_name: {"bandgap_eV": float, "CBM": float,
                                  "VBM": float, "is_direct": bool}}
    """
    results = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV appears empty or missing header: {csv_path}")

        # Accept common naming variants across pipelines.
        id_field_candidates = ("folder", "MOF", "name", "cif_id", "id")
        id_field = next((c for c in id_field_candidates if c in reader.fieldnames), None)
        if id_field is None:
            raise KeyError(
                f"Could not find MOF identifier column in {csv_path}. "
                f"Tried {id_field_candidates}; found columns={reader.fieldnames}"
            )

        for row in reader:
            name = row.get(id_field, "").strip()
            if not name:
                continue
            bg = row.get("bandgap_eV", "").strip()
            try:
                bg_val = float(bg)
            except (ValueError, TypeError):
                bg_val = float("inf")
            results[name] = {
                "bandgap_eV": bg_val,
                "CBM": float(row.get("CBM", 0) or 0),
                "VBM": float(row.get("VBM", 0) or 0),
                "is_direct": row.get("is_direct", "").strip() == "True",
            }
    print(f"    Loaded {len(results)} entries from bandgap CSV")
    return results


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
#  Panel (a): Overview — nominations colored by DFT bandgap
# ──────────────────────────────────────────────────────────────────────
def panel_overview(coords, is_labeled, bandgaps_labeled, threshold,
                   nom_indices, nom_bandgaps, nom_names,
                   output_dir, exp_label, filename_prefix):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Background: unlabeled gray, labeled blue
    unlabeled_mask = ~is_labeled
    ax.scatter(coords[unlabeled_mask, 0], coords[unlabeled_mask, 1],
               c="#c0c0c0", s=3, alpha=0.45, rasterized=True, zorder=1,
               label=f"Unlabeled ({int(unlabeled_mask.sum()):,})")
    ax.scatter(coords[is_labeled, 0], coords[is_labeled, 1],
               c="#2171b5", s=4, alpha=0.50, rasterized=True, zorder=2,
               label=f"DFT-labeled ({int(is_labeled.sum()):,})")

    # Known positives
    pos_mask = is_labeled & (bandgaps_labeled < threshold)
    n_pos = int(pos_mask.sum())
    if n_pos > 0:
        ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1],
                   c="#2171b5", s=12, marker="^",
                   edgecolors="black", linewidths=0.4,
                   zorder=3, alpha=0.85)

    # Nominations: show ONLY hits as stars (cleaner visual)
    nom_coords = coords[nom_indices]
    hit_mask = np.isfinite(nom_bandgaps) & (nom_bandgaps < threshold)
    n_hit = int(hit_mask.sum())
    if n_hit > 0:
        ax.scatter(nom_coords[hit_mask, 0], nom_coords[hit_mask, 1],
                   c="#f1c40f", s=155, marker="*",
                   edgecolors="black", linewidths=0.6,
                   zorder=6, alpha=0.98)

    # Legend
    _leg = [
        Line2D([], [], marker="o", color="w", markerfacecolor="#2171b5",
               markersize=5, label=f"DFT-labeled ({int(is_labeled.sum()):,})"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#c0c0c0",
               markersize=5, label=f"Unlabeled ({int(unlabeled_mask.sum()):,})"),
    ]
    if n_pos > 0:
        _leg.append(Line2D([], [], marker="^", color="w",
                           markerfacecolor="#2171b5", markeredgecolor="black",
                           markeredgewidth=0.4, markersize=6,
                           label=f"Known positives ({n_pos})"))
    _leg.append(Line2D([], [], marker="*", color="w",
                       markerfacecolor="#f1c40f", markeredgecolor="black",
                       markeredgewidth=0.5, markersize=8,
                       label=f"Nominated hits < {threshold:.1f} eV ({n_hit})"))

    ax.legend(handles=_leg, loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.95, borderpad=0.5,
              handletextpad=0.5, handlelength=1.4, fontsize=7,
              markerscale=0.8)
    ax.set_title(f"Nominated structures — hits only  [{exp_label}]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, f"{filename_prefix}_overview")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
#  Panel (b): Annotated nominations with full background
# ──────────────────────────────────────────────────────────────────────
def panel_annotated(coords, is_labeled, nom_indices, nom_bandgaps,
                    nom_names, nom_is_direct, threshold, output_dir, exp_label,
                    filename_prefix):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Background: all MOFs as context
    unlabeled_mask = ~is_labeled
    ax.scatter(coords[unlabeled_mask, 0], coords[unlabeled_mask, 1],
               c="#c0c0c0", s=2.2, alpha=0.35, rasterized=True, zorder=1)
    ax.scatter(coords[is_labeled, 0], coords[is_labeled, 1],
               c="#2171b5", s=2.8, alpha=0.42, rasterized=True, zorder=2)

    nom_coords = coords[nom_indices]
    hit_mask = np.isfinite(nom_bandgaps) & (nom_bandgaps < threshold)
    if int(hit_mask.sum()) > 0:
        ax.scatter(nom_coords[hit_mask, 0], nom_coords[hit_mask, 1],
                   c="#f1c40f", s=200, marker="*",
                   edgecolors="black", linewidths=0.75,
                   zorder=6, alpha=0.98)

    # Annotate ONLY hits with name + bandgap (cleaner panel)
    offsets = [
        (8, 8), (8, -10), (-10, 8), (-10, -10),
        (14, 2), (2, 14), (-14, 2), (2, -14),
    ]
    hit_indices = np.where(hit_mask)[0]
    hit_coords = nom_coords[hit_indices]
    order_local = np.argsort(hit_coords[:, 0] + 0.35 * hit_coords[:, 1]) if len(hit_coords) else []
    for rank, local_idx in enumerate(order_local):
        k = hit_indices[local_idx]
        short = nom_names[k].replace("_FSR", "")
        bg = nom_bandgaps[k]
        direct_str = " (D)" if nom_is_direct[k] else ""
        if np.isfinite(bg):
            label = f"{short}\n{bg:.3f} eV{direct_str}"
        else:
            label = f"{short}\ninf"
        dx, dy = offsets[rank % len(offsets)]
        ax.annotate(label, (nom_coords[k, 0], nom_coords[k, 1]),
                    fontsize=5.2, fontweight="bold",
                    xytext=(dx, dy), textcoords="offset points",
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="0.6", alpha=0.9, lw=0.35),
                    arrowprops=dict(arrowstyle="-", color="0.45", lw=0.35))

    ax.set_title(f"Nominated MOFs — annotated hits only (< {threshold:.1f} eV)  [{exp_label}]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, f"{filename_prefix}_annotated")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
#  Process one experiment
# ──────────────────────────────────────────────────────────────────────
def process_experiment(npz_path, exp_name, exp_label,
                       labeled_bg, nom_results, args):
    print(f"\n{'─' * 60}")
    print(f"  Processing {exp_name}")
    print(f"  NPZ: {npz_path}")
    print(f"{'─' * 60}")

    if not os.path.exists(npz_path):
        print(f"  *** ERROR: NPZ not found: {npz_path}")
        return

    cids, embs, bgs_raw, splits_raw, is_lab_raw = load_npz(npz_path)

    # Rebuild is_labeled
    labeled_cid_set = set(labeled_bg.keys())
    if is_lab_raw is not None:
        is_labeled = is_lab_raw
    else:
        is_labeled = np.array([c in labeled_cid_set for c in cids])

    bandgaps = np.array([labeled_bg.get(c, float(bgs_raw[i]))
                         for i, c in enumerate(cids)], dtype=float)

    # Match nominations to embedding indices
    nom_lookup = {c: v for c, v in nom_results.items()}
    nom_indices = []
    nom_bandgaps = []
    nom_names = []
    nom_is_direct = []
    matched_names = []

    for i, cid in enumerate(cids):
        result = _flex_lookup(cid, nom_lookup)
        if result is not None:
            nom_indices.append(i)
            nom_bandgaps.append(result["bandgap_eV"])
            nom_names.append(cid)
            nom_is_direct.append(result["is_direct"])
            matched_names.append(cid)

    nom_indices = np.array(nom_indices, dtype=int)
    nom_bandgaps = np.array(nom_bandgaps, dtype=float)

    print(f"    {len(cids)} MOFs ({int(is_labeled.sum())} labeled, "
          f"{len(cids) - int(is_labeled.sum())} unlabeled)")
    print(f"    Nominations matched: {len(nom_indices)}/{len(nom_results)}")

    if len(nom_indices) == 0:
        print("  *** WARNING: No nominations matched any embedding ID! ***")
        print("    Nomination names:", list(nom_results.keys())[:5], "...")
        print("    Embedding IDs sample:", cids[:5], "...")
        return

    # Print matched nominations
    for k in range(len(nom_indices)):
        bg = nom_bandgaps[k]
        d = "direct" if nom_is_direct[k] else "indirect"
        hit = np.isfinite(bg) and (bg < args.threshold)
        tag = "HIT(filled star)" if hit else "NON-HIT(outlined star)"
        print(f"      {nom_names[k]:>20s}  →  {bg:.4f} eV  ({d}, {tag})")

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
        # Also check experiment's own UMAP cache
        exp_dir = os.path.dirname(npz_path)
        exp_umap_cache = os.path.join(exp_dir, "posttrain_umap_cache.npz")
        if os.path.exists(exp_umap_cache):
            print(f"    Found experiment UMAP cache: {exp_umap_cache}")
            cache = np.load(exp_umap_cache, allow_pickle=True)
            cached_ids = [str(x) for x in cache["cif_ids"]]
            if cached_ids == cids:
                coords = cache["coords"]
                print(f"    {coords.shape[0]} cached points from experiment dir")
            else:
                print(f"    WARNING: cached IDs differ — recomputing UMAP")

    if coords is None:
        print(f"    Computing UMAP ...")
        coords = compute_umap(embs, args.n_neighbors, args.min_dist,
                              seed=args.seed)

    if args.save_umap_cache:
        np.savez_compressed(cache_path, coords=coords,
                            cif_ids=np.array(cids))
        print(f"    UMAP cache saved: {cache_path}")

    # Filter invalid coords
    valid = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    norms = np.linalg.norm(embs, axis=1)
    valid &= norms > 1e-6
    n_skip = int((~valid).sum())
    if n_skip > 0:
        print(f"    Skipping {n_skip} MOFs with invalid embeddings/coords")
        # Reindex: map old indices to new
        old_to_new = np.full(len(cids), -1, dtype=int)
        new_idx = 0
        for i in range(len(cids)):
            if valid[i]:
                old_to_new[i] = new_idx
                new_idx += 1
        # Remap nom_indices
        new_nom_indices = []
        new_nom_bandgaps = []
        new_nom_names = []
        new_nom_is_direct = []
        for k in range(len(nom_indices)):
            new_i = old_to_new[nom_indices[k]]
            if new_i >= 0:
                new_nom_indices.append(new_i)
                new_nom_bandgaps.append(nom_bandgaps[k])
                new_nom_names.append(nom_names[k])
                new_nom_is_direct.append(nom_is_direct[k])
        nom_indices = np.array(new_nom_indices, dtype=int)
        nom_bandgaps = np.array(new_nom_bandgaps, dtype=float)
        nom_names = new_nom_names
        nom_is_direct = new_nom_is_direct

        coords = coords[valid]
        is_labeled = is_labeled[valid]
        bandgaps = bandgaps[valid]

    prefix = f"nominated_{exp_name}"

    # Panel (a): overview
    panel_overview(coords, is_labeled, bandgaps, args.threshold,
                   nom_indices, nom_bandgaps, nom_names,
                   args.output_dir, exp_label, prefix)

    # Panel (b): annotated with background
    panel_annotated(coords, is_labeled, nom_indices, nom_bandgaps,
                    nom_names, nom_is_direct, args.threshold,
                    args.output_dir, exp_label,
                    prefix)

    # Summary JSON
    summary = {
        "experiment": exp_name,
        "total_mofs": int(coords.shape[0]),
        "labeled": int(is_labeled.sum()),
        "unlabeled": int(coords.shape[0]) - int(is_labeled.sum()),
        "nominations_matched": len(nom_indices),
        "nominations_total": len(nom_results),
        "nominations": [
            {"name": nom_names[k],
             "bandgap_eV": float(nom_bandgaps[k]),
             "hit": bool(np.isfinite(nom_bandgaps[k]) and
                         (nom_bandgaps[k] < args.threshold)),
             "is_direct": bool(nom_is_direct[k]),
             "umap_x": float(coords[nom_indices[k], 0]),
             "umap_y": float(coords[nom_indices[k], 1])}
            for k in range(len(nom_indices))
        ],
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
        description="UMAP of pretrained PMTransformer embeddings with "
                    "nominated structures colored by DFT bandgap",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    pa.add_argument("--pretrained_npz", required=True,
                    help="Path to all_embeddings.npz from pretrained "
                         "(non-finetuned) PMTransformer forward pass")
    pa.add_argument("--finetuned_npz", default=None,
                    help="Optional: path to posttrain_embeddings.npz from a "
                         "fine-tuned model (generates a second set of plots)")
    pa.add_argument("--finetuned_name", default="finetuned",
                    help="Short label for the fine-tuned model (e.g. exp370)")
    pa.add_argument("--bandgap_csv", required=True,
                    help="Path to bandgap_results.csv from mass_bandgap_analyzer.py")
    pa.add_argument("--qmof_csv", required=True,
                    help="Path to qmof.csv (authoritative metal-node source)")
    pa.add_argument("--labeled_splits_dir", required=True,
                    help="Dir with {train,val,test}_bandgaps_regression.json")
    pa.add_argument("--output_dir", default="./nominated_umap_figures")
    pa.add_argument("--threshold", type=float, default=BANDGAP_THRESHOLD,
                    help="Bandgap threshold for positive class (eV)")
    pa.add_argument("--n_neighbors", type=int, default=30)
    pa.add_argument("--min_dist", type=float, default=0.3)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--load_umap_cache", action="store_true")
    pa.add_argument("--save_umap_cache", action="store_true")

    args = pa.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_publication_style()

    print("=" * 70)
    print("  NOMINATED STRUCTURES UMAP — Pretrained PMTransformer + DFT Bandgap")
    print("=" * 70)

    # Load labels
    print("\n[1] Loading labeled splits ...")
    labeled_bg = load_split_labels(os.path.abspath(args.labeled_splits_dir))
    print(f"    Total labeled MOFs: {len(labeled_bg)}")

    # Load nomination bandgaps
    print("\n[2] Loading DFT bandgap results ...")
    nom_results = load_bandgap_csv(os.path.abspath(args.bandgap_csv))
    nom_metals = get_metals_from_qmof_csv(list(nom_results.keys()),
                                          os.path.abspath(args.qmof_csv))
    for name in nom_results:
        nom_results[name]["metal_nodes"] = nom_metals.get(name, "Unknown")
    for name, info in nom_results.items():
        bg = info["bandgap_eV"]
        d = "direct" if info["is_direct"] else "indirect"
        m = info["metal_nodes"]
        print(f"    {name:>20s}:  {bg:.4f} eV  ({d}, metal={m})")

    # ── Pretrained (foundational) model ──
    print("\n[3] Generating plots for pretrained PMTransformer ...")
    process_experiment(
        os.path.abspath(args.pretrained_npz),
        "pretrained", "pretrained PMTransformer",
        labeled_bg, nom_results, args)

    # ── Optional: fine-tuned model ──
    if args.finetuned_npz is not None:
        print(f"\n[4] Generating plots for fine-tuned model "
              f"({args.finetuned_name}) ...")
        process_experiment(
            os.path.abspath(args.finetuned_npz),
            args.finetuned_name,
            f"fine-tuned {args.finetuned_name}",
            labeled_bg, nom_results, args)

    print(f"\n{'=' * 70}")
    print(f"  Done.  All outputs in {args.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
