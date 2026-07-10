#!/usr/bin/env python3
"""
SOAP-Based UMAP Chemical Space — NN-Independent Structural Landscape
=====================================================================

Same 2D UMAP visualization as Figure 2, but using SOAP descriptors
(computed purely from crystal geometry) instead of NN embeddings.

If the SOAP UMAP looks structurally similar to the NN UMAP, it proves
the NN captured real structural information — killing the circularity
argument in one visual.

Panels (each saved as separate PNG + SVG + PDF):
  (a) Labeled vs. unlabeled  — same coloring as Figure 2a
  (b) DFT bandgap            — viridis colormap, red circles for positives
  (c) Train / val / test     — split assignments
  (d) Ensemble nominations   — top-25 DFT candidates highlighted

Pipeline:
  1. Compute SOAP descriptors from CIF files (cached after first run)
  2. Run UMAP on SOAP vectors → 2D coordinates
  3. Generate publication-quality panels

Usage:
  python figures/soap_descriptors_umap.py \\
      --cif_dir data/raw/cif \\
      --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --unlabeled_json data/unlabeled/test_bandgaps_regression.json \\
      --output_dir figures_output/soap_umap

  # With ensemble nominations:
  python soap_descriptors_umap.py \\
      --cif_dir ... --labeled_splits_dir ... --unlabeled_json ... \\
      --nominations data/unlabeled/nomination-SOAP/FINAL_TOP25_diverse.txt \\
      --output_dir figures_output/soap_umap

  # Reuse cached SOAP (fast re-runs for tweaking plots):
  python figures/soap_descriptors_umap.py \\
      --cif_dir data/raw/cif --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --unlabeled_json data/unlabeled/test_bandgaps_regression.json \\
      --soap_cache figures_output/soap_umap/soap_descriptors.npz \\
      --output_dir figures_output/soap_umap

Requirements:  pip install dscribe ase numpy matplotlib umap-learn scipy
"""

import os
import sys
import json
import re
import time
import argparse
import numpy as np
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────
#  SOAP parameters
# ──────────────────────────────────────────────────────────────────────
SOAP_RCUT = 6.0
SOAP_NMAX = 4
SOAP_LMAX = 4
SOAP_SIGMA = 0.5
SOAP_PERIODIC = True

# ──────────────────────────────────────────────────────────────────────
#  Publication style
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


# ──────────────────────────────────────────────────────────────────────
#  Save helper
# ──────────────────────────────────────────────────────────────────────
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
#  SOAP computation (same as soap_validation.py, self-contained)
# ──────────────────────────────────────────────────────────────────────
def discover_cif_files(cif_dir):
    cifs = {}
    for fn in os.listdir(cif_dir):
        if fn.endswith(".cif"):
            cifs[fn[:-4]] = os.path.join(cif_dir, fn)
    return cifs


def compute_soap_descriptors(cif_dir, cif_ids, output_path):
    from ase.io import read as ase_read
    from dscribe.descriptors import SOAP

    print("    Reading CIF files and collecting species ...")
    cif_map = discover_cif_files(cif_dir)
    structures = {}
    all_species = set()
    n_skip = 0
    skipped = []

    t0 = time.time()
    for i, cid in enumerate(cif_ids):
        if cid not in cif_map:
            n_skip += 1
            if len(skipped) < 20:
                skipped.append(cid)
            continue
        try:
            atoms = ase_read(cif_map[cid])
            all_species.update(atoms.get_chemical_symbols())
            structures[cid] = atoms
        except Exception as e:
            n_skip += 1
            if len(skipped) < 20:
                skipped.append(f"{cid}({e.__class__.__name__})")
        if (i + 1) % 2000 == 0:
            print(f"      Read {i+1}/{len(cif_ids)} CIFs ...")

    elapsed = time.time() - t0
    print(f"    Read {len(structures)} CIF files in {elapsed:.0f}s  "
          f"(skipped {n_skip})")
    if skipped:
        print(f"    Skipped (sample): {skipped[:10]}")

    species = sorted(all_species)
    print(f"    Unique species: {len(species)}  "
          f"{species[:20]}{'...' if len(species) > 20 else ''}")

    soap = SOAP(
        species=species, r_cut=SOAP_RCUT, n_max=SOAP_NMAX,
        l_max=SOAP_LMAX, sigma=SOAP_SIGMA,
        periodic=SOAP_PERIODIC, average="inner", sparse=False,
    )
    print(f"    SOAP descriptor dimension: {soap.get_number_of_features()}")
    print("    Computing SOAP descriptors ...")

    ordered_ids, soap_list = [], []
    t0 = time.time()
    last_report = t0

    for idx, cid in enumerate(cif_ids):
        if cid not in structures:
            continue
        try:
            desc = soap.create(structures[cid])
            if desc.ndim == 2:
                desc = desc[0]
            soap_list.append(desc)
            ordered_ids.append(cid)
        except Exception:
            continue

        now = time.time()
        if now - last_report >= 30:
            last_report = now
            done = len(soap_list)
            total = len(structures)
            elapsed_s = now - t0
            rate = done / elapsed_s if elapsed_s > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(remaining), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            print(f"      [{done:>6}/{total}]  "
                  f"{done / total * 100:5.1f}%  "
                  f"ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s  "
                  f"({rate:.1f} MOF/s)", flush=True)

    soap_matrix = np.array(soap_list, dtype=np.float32)
    elapsed_total = time.time() - t0
    em, es = divmod(int(elapsed_total), 60)
    eh, em = divmod(em, 60)
    print(f"    SOAP done: {len(ordered_ids)} MOFs, dim={soap_matrix.shape[1]}  "
          f"[{eh}h{em:02d}m{es:02d}s]")

    np.savez_compressed(
        output_path,
        cif_ids=np.array(ordered_ids),
        soap_descriptors=soap_matrix,
        species=np.array(species),
        params=np.array([SOAP_RCUT, SOAP_NMAX, SOAP_LMAX, SOAP_SIGMA]),
    )
    print(f"    SOAP cache saved -> {output_path}")

    # Also save sparse cache for memory-safe future loads
    from scipy.sparse import csr_matrix as _csr
    soap_sparse = _csr(soap_matrix)
    del soap_matrix
    sparse_path = output_path.replace(".npz", "_sparse.npz")
    np.savez_compressed(
        sparse_path,
        cif_ids=np.array(ordered_ids),
        sp_data=soap_sparse.data,
        sp_indices=soap_sparse.indices,
        sp_indptr=soap_sparse.indptr,
        sp_shape=np.array(soap_sparse.shape),
    )
    print(f"    Sparse cache saved -> {sparse_path}")
    return ordered_ids, soap_sparse


def _read_npy_from_zip(zip_path, array_name):
    """Read a single .npy array from inside a .npz (zip) archive using
    streaming — never loads the full decompressed array into RAM.
    Returns a generator yielding one row (1-D array) at a time."""
    import zipfile
    import struct

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # .npz stores arrays as <name>.npy inside the zip
        npy_name = array_name + ".npy"
        with zf.open(npy_name) as f:
            # Parse .npy header (version 1.0 or 2.0)
            magic = f.read(6)
            assert magic[:6] == b'\x93NUMPY', f"Bad .npy magic in {npy_name}"
            major, minor = struct.unpack('BB', f.read(2))
            if major == 1:
                header_len = struct.unpack('<H', f.read(2))[0]
            else:
                header_len = struct.unpack('<I', f.read(4))[0]
            header = f.read(header_len).decode('latin1')

            # Parse header dict (safe: it's a Python literal)
            import ast
            header_dict = ast.literal_eval(header.strip())
            shape = header_dict['shape']
            dtype = np.dtype(header_dict['descr'])
            order = header_dict.get('fortran_order', False)
            assert not order, "Fortran-ordered arrays not supported"

            n_rows, n_cols = shape
            row_bytes = n_cols * dtype.itemsize

            # Yield rows one at a time — peak RAM = 1 row (~1 MB)
            for _ in range(n_rows):
                raw = f.read(row_bytes)
                if len(raw) < row_bytes:
                    break
                yield np.frombuffer(raw, dtype=dtype)


def load_soap_cache(cache_path):
    """Load SOAP cache.  Tries the sparse .npz first (fast, low-memory).
    Falls back to the legacy dense .npz by streaming rows from the zip
    archive to build a sparse matrix — peak RAM stays ~200 MB, not ~19 GB."""
    import scipy.sparse as sp

    # Prefer sparse cache if it exists
    sparse_path = cache_path.replace(".npz", "_sparse.npz")
    if os.path.exists(sparse_path):
        print(f"    Loading sparse SOAP cache: {sparse_path}")
        data = np.load(sparse_path, allow_pickle=True)
        cif_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                   for x in data["cif_ids"]]
        soap_matrix = sp.csr_matrix(
            (data["sp_data"], data["sp_indices"], data["sp_indptr"]),
            shape=tuple(data["sp_shape"]))
        sparse_mb = (soap_matrix.data.nbytes + soap_matrix.indices.nbytes +
                     soap_matrix.indptr.nbytes) / 1e6
        print(f"    {len(cif_ids)} MOFs, dim={soap_matrix.shape[1]}, "
              f"{sparse_mb:.0f} MB sparse")
        return cif_ids, soap_matrix

    # Fallback: legacy dense npz — stream rows from zip to avoid RAM spike
    print(f"    Dense SOAP cache: {cache_path}")
    print(f"    Streaming rows from zip to build sparse matrix ...")

    # Load cif_ids (small array, fine to load fully)
    data = np.load(cache_path, allow_pickle=True)
    cif_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
               for x in data["cif_ids"]]
    # Get shape without loading the full array
    soap_shape = None
    import zipfile
    with zipfile.ZipFile(cache_path, 'r') as zf:
        with zf.open("soap_descriptors.npy") as f:
            import struct
            magic = f.read(6)
            major, _ = struct.unpack('BB', f.read(2))
            if major == 1:
                hl = struct.unpack('<H', f.read(2))[0]
            else:
                hl = struct.unpack('<I', f.read(4))[0]
            import ast
            soap_shape = ast.literal_eval(
                f.read(hl).decode('latin1').strip())['shape']
    n, d = soap_shape
    print(f"    {n} MOFs, dim={d}")
    del data   # release the NpzFile handle

    # Build sparse row-by-row from zip stream
    from scipy.sparse import lil_matrix
    soap_lil = lil_matrix((n, d), dtype=np.float32)
    for i, row in enumerate(_read_npy_from_zip(cache_path, "soap_descriptors")):
        nz = np.nonzero(row)[0]
        if len(nz):
            soap_lil[i, nz] = row[nz]
        if (i + 1) % 5000 == 0:
            print(f"      {i+1}/{n} rows converted ...")
    soap_matrix = soap_lil.tocsr()
    del soap_lil

    sparse_mb = (soap_matrix.data.nbytes + soap_matrix.indices.nbytes +
                 soap_matrix.indptr.nbytes) / 1e6
    nnz_pct = 100 * soap_matrix.nnz / (n * d)
    print(f"    Sparse: {sparse_mb:.0f} MB  ({nnz_pct:.1f}% non-zero)")

    # Save sparse cache for next time (instant loads)
    print(f"    Saving sparse cache for future runs: {sparse_path}")
    np.savez_compressed(
        sparse_path,
        cif_ids=np.array(cif_ids),
        sp_data=soap_matrix.data,
        sp_indices=soap_matrix.indices,
        sp_indptr=soap_matrix.indptr,
        sp_shape=np.array(soap_matrix.shape),
    )
    print(f"    Sparse cache saved -> {sparse_path}")

    return cif_ids, soap_matrix


# ──────────────────────────────────────────────────────────────────────
#  Metadata loading
# ──────────────────────────────────────────────────────────────────────
def load_split_labels(splits_dir):
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
        print(f"    {split_name}: {len(d)} MOFs")
    if not found_any:
        print(f"  *** WARNING: No split JSONs found in {splits_dir}")
        print(f"  *** Expected files: train_bandgaps_regression.json, etc.")
        print(f"  *** Directory contents: {os.listdir(splits_dir)[:20]}")
    return labels, assignments


def load_unlabeled_ids(json_path):
    with open(json_path) as fh:
        return set(json.load(fh).keys())


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


def load_nominations(filepath):
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
def compute_umap(soap_matrix, n_neighbors=30, min_dist=0.3, seed=42):
    try:
        from umap import UMAP
    except ImportError:
        sys.exit("ERROR: umap-learn not installed.  pip install umap-learn")

    n, d = soap_matrix.shape
    print(f"    {n} points, dim={d} ...")

    # UMAP handles scipy sparse matrices natively with cosine metric.
    # No dimensionality reduction needed — the natural sparsity of SOAP
    # (~99% zeros) keeps memory tractable.
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                   metric="cosine", random_state=seed, n_jobs=-1,
                   low_memory=True)
    coords = reducer.fit_transform(soap_matrix)
    print(f"    UMAP done -> shape {coords.shape}")
    return coords


# ══════════════════════════════════════════════════════════════════════
#  Panel (a): Labeled vs Unlabeled
# ══════════════════════════════════════════════════════════════════════
def panel_a_labeled_unlabeled(coords, is_labeled, output_dir):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
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
    ax.set_title("(a)  Labeled vs. unlabeled  [SOAP space]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "soap_umap_a_labeled_unlabeled")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  Panel (b): DFT bandgap — discrete bins like reference paper
# ══════════════════════════════════════════════════════════════════════
BANDGAP_BINS = [
    (0.0, 0.5,  "#d73027",  "0 – 0.5 eV"),    # deep red  (metals / tiny gap)
    (0.5, 1.0,  "#fc8d59",  "0.5 – 1 eV"),     # orange    (narrow gap)
    (1.0, 2.0,  "#fee08b",  "1 – 2 eV"),       # yellow
    (2.0, 3.0,  "#d9ef8b",  "2 – 3 eV"),       # light green
    (3.0, 4.0,  "#91bfdb",  "3 – 4 eV"),       # light blue
    (4.0, 99.0, "#4575b4",  "≥ 4 eV"),         # deep blue (wide gap)
]


def panel_b_bandgap(coords, is_labeled, bandgaps, threshold, output_dir):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
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
    ax.set_title("(b)  DFT bandgap  [SOAP space]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "soap_umap_b_bandgap")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  Panel (c): Train / Val / Test splits
# ══════════════════════════════════════════════════════════════════════
def panel_c_splits(coords, is_labeled, bandgaps, split_labels,
                   threshold, output_dir):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    lab_coords = coords[is_labeled]
    lab_bgs    = bandgaps[is_labeled]
    lab_splits = split_labels[is_labeled]
    pos_mask   = lab_bgs < threshold

    colors_map = {"train": "#4292c6", "val": "#fd8d3c", "test": "#969696"}
    order = ["test", "val", "train"]

    for sp in order:
        mask = lab_splits == sp
        n = int(mask.sum())
        ax.scatter(lab_coords[mask, 0], lab_coords[mask, 1],
                   c=colors_map.get(sp, "#cccccc"), s=2.5, alpha=0.40,
                   rasterized=True, zorder=1 if sp == "test" else 2,
                   label=f"{sp.capitalize()} ({n:,})")

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
    ax.set_title("(c)  Train / val / test  [SOAP space]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "soap_umap_c_splits")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  Panel (d): Ensemble top nominations
# ══════════════════════════════════════════════════════════════════════
def panel_d_nominations(coords, is_labeled, bandgaps, threshold,
                        nomination_mask, nomination_cids, output_dir):
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
    ax.set_title("(d)  Ensemble nominations  [SOAP space]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "soap_umap_d_ensemble")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(
        description="SOAP-based UMAP Chemical Space Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    pa.add_argument("--cif_dir", required=True,
                    help="Directory with .cif files for all MOFs")
    pa.add_argument("--labeled_splits_dir", required=True,
                    help="Dir with {train,val,test}_bandgaps_regression.json")
    pa.add_argument("--unlabeled_json", default=None,
                    help="Unlabeled test_bandgaps_regression.json (unlabeled IDs)")
    pa.add_argument("--nominations", default=None,
                    help="File with top-K ensemble CIF IDs (one per line)")
    pa.add_argument("--soap_cache", default=None,
                    help="Pre-computed soap_descriptors.npz (skip SOAP stage)")
    pa.add_argument("--output_dir", default="./figures_output/soap_umap")
    pa.add_argument("--threshold", type=float, default=1.0)
    pa.add_argument("--n_neighbors", type=int, default=30)
    pa.add_argument("--min_dist", type=float, default=0.3)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--recompute_soap", action="store_true",
                    help="Force re-computation of SOAP (ignore cache)")
    pa.add_argument("--load_umap_cache", default=None,
                    help="NPZ with cached SOAP-UMAP coords")
    pa.add_argument("--save_umap_cache", action="store_true",
                    help="Save UMAP coords for fast re-runs")

    args = pa.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_publication_style()

    print("=" * 70)
    print("  SOAP-BASED UMAP — NN-Independent Chemical Space")
    print("=" * 70)

    # ── 1. Load labels ────────────────────────────────────────────────
    print("\n[1/4] Loading labels ...")
    labeled_bg, labeled_splits = load_split_labels(args.labeled_splits_dir)
    labeled_cids = set(labeled_bg.keys())
    print(f"    Total labeled: {len(labeled_cids)}")

    unlabeled_cids = set()
    if args.unlabeled_json and os.path.exists(args.unlabeled_json):
        unlabeled_cids = load_unlabeled_ids(args.unlabeled_json)
        print(f"    Total unlabeled: {len(unlabeled_cids)}")

    # All CIF IDs we care about
    all_cids = sorted(labeled_cids | unlabeled_cids)
    print(f"    Combined: {len(all_cids)} MOFs")

    # Ensemble nominations
    nomination_cid_list = []
    if args.nominations and os.path.exists(args.nominations):
        nomination_cid_list = load_nominations(args.nominations)

    # ── 2. SOAP descriptors ───────────────────────────────────────────
    print(f"\n[2/4] SOAP descriptors ...")
    soap_cache_path = os.path.join(args.output_dir, "soap_descriptors.npz")

    if args.soap_cache and os.path.exists(args.soap_cache):
        print(f"    Using external cache: {args.soap_cache}")
        soap_cids, soap_matrix = load_soap_cache(args.soap_cache)
    elif os.path.exists(soap_cache_path) and not args.recompute_soap:
        print(f"    Cache found: {soap_cache_path}")
        soap_cids, soap_matrix = load_soap_cache(soap_cache_path)
    else:
        print(f"    Computing from CIFs in {args.cif_dir} ...")
        soap_cids, soap_matrix = compute_soap_descriptors(
            args.cif_dir, all_cids, soap_cache_path)

    # ── Match SOAP IDs to labels (flexible: ±_FSR, ±.cif) ─────────
    soap_sub = soap_matrix          # use all SOAP data
    present_cids = list(soap_cids)  # keep variable for UMAP cache compat
    n = len(soap_cids)
    is_labeled   = np.zeros(n, dtype=bool)
    bandgaps     = np.full(n, np.nan, dtype=float)
    split_labels = np.array(["unlabeled"] * n)

    for i, cid in enumerate(soap_cids):
        bg = _flex_lookup(cid, labeled_bg)
        if bg is not None:
            is_labeled[i] = True
            bandgaps[i] = bg
            sp = _flex_lookup(cid, labeled_splits)
            if sp is not None:
                split_labels[i] = sp

    # Ensemble nomination mask (flexible matching)
    nomination_lookup = {c: True for c in nomination_cid_list}
    nomination_mask = np.array([_flex_lookup(c, nomination_lookup) is not None
                            for c in soap_cids])
    # For annotation: keep order aligned with mask
    nominations_present = [c for c in soap_cids
                      if _flex_lookup(c, nomination_lookup) is not None]

    n_lab = int(is_labeled.sum())
    n_ulab = n - n_lab
    print(f"\n    Final: {n} MOFs ({n_lab} labeled, {n_ulab} unlabeled)")
    if nominations_present:
        print(f"    Ensemble nominations in SOAP: {len(nominations_present)}/{len(nomination_cid_list)}")

    # Positive counts per split
    for sp in ("train", "val", "test"):
        sp_mask = split_labels == sp
        sp_pos = sp_mask & (bandgaps < args.threshold)
        print(f"    {sp}: {int(sp_mask.sum())} MOFs, "
              f"{int(sp_pos.sum())} positives (< {args.threshold} eV)")

    # ── 3. UMAP on SOAP ──────────────────────────────────────────────
    if args.load_umap_cache and os.path.exists(args.load_umap_cache):
        print(f"\n[3/4] Loading cached SOAP-UMAP coords: {args.load_umap_cache}")
        cache = np.load(args.load_umap_cache, allow_pickle=True)
        coords = cache["coords"]
        cached_ids = [str(x) for x in cache["cif_ids"]]
        if cached_ids == present_cids:
            print(f"    {coords.shape[0]} cached points loaded")
        else:
            print("    WARNING: cached IDs differ — recomputing UMAP")
            coords = compute_umap(soap_sub, args.n_neighbors, args.min_dist,
                                  seed=args.seed)
    else:
        print(f"\n[3/4] Computing UMAP on SOAP descriptors ...")
        coords = compute_umap(soap_sub, args.n_neighbors, args.min_dist,
                              seed=args.seed)

    if args.save_umap_cache:
        cp = os.path.join(args.output_dir, "soap_umap_cache.npz")
        np.savez_compressed(cp, coords=coords,
                            cif_ids=np.array(present_cids))
        print(f"    UMAP cache saved: {cp}")

    # ── 4. Generate panels ────────────────────────────────────────────
    print(f"\n[4/4] Generating panels ...")

    panel_a_labeled_unlabeled(coords, is_labeled, args.output_dir)
    panel_b_bandgap(coords, is_labeled, bandgaps, args.threshold,
                    args.output_dir)
    panel_c_splits(coords, is_labeled, bandgaps, split_labels,
                   args.threshold, args.output_dir)

    if nominations_present:
        panel_d_nominations(coords, is_labeled, bandgaps, args.threshold,
                       nomination_mask, nominations_present, args.output_dir)
    else:
        print("    Skipping panel (d) — no ensemble nominations provided")

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "total_mofs": n,
        "labeled": n_lab,
        "unlabeled": n_ulab,
        "soap_dim": int(soap_sub.shape[1]),
        "umap_params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
            "seed": args.seed,
        },
        "soap_params": {
            "r_cut": SOAP_RCUT, "n_max": SOAP_NMAX,
            "l_max": SOAP_LMAX, "sigma": SOAP_SIGMA,
        },
        "ensemble_nominations_found": len(nominations_present),
    }
    sp = os.path.join(args.output_dir, "soap_umap_summary.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Done.  All outputs in {args.output_dir}/")
    print(f"  Panels: (a) labeled/unlabeled, (b) bandgap, (c) splits"
          + (", (d) ensemble nominations" if nominations_present else ""))
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
