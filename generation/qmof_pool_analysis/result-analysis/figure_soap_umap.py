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

Usage (cluster):
  python figure_soap_umap.py \\
      --cif_dir /path/to/preprocessed_structures/test \\
      --labeled_splits_dir /path/to/data/splits/strategy_d_farthest_point \\
      --unlabeled_json /path/to/unlabeled_pool/test_bandgaps_regression.json \\
      --output_dir ./soap_umap_figures

  # With ensemble nominations:
  python figure_soap_umap.py \\
      --cif_dir ... --labeled_splits_dir ... --unlabeled_json ... \\
      --nominated_top_predictions /path/to/FINAL_DFT_TOP25.txt \\
      --output_dir ./soap_umap_figures

  # Reuse cached SOAP (fast re-runs for tweaking plots):
  python figure_soap_umap.py \\
      --cif_dir ... --labeled_splits_dir ... --unlabeled_json ... \\
      --soap_cache ./soap_analysis/soap_descriptors.npz \\
      --output_dir ./soap_umap_figures

Requirements:  pip install dscribe ase numpy matplotlib umap-learn scipy
"""

import os
import sys
import json
import csv
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
#  SOAP computation (same as figure_soap_analysis.py, self-contained)
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

    # If the provided path itself is a sparse cache, load it directly.
    data = np.load(cache_path, allow_pickle=True)
    sparse_keys = {"sp_data", "sp_indices", "sp_indptr", "sp_shape"}
    if sparse_keys.issubset(set(data.files)):
        print(f"    Loading sparse SOAP cache: {cache_path}")
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

    # Prefer sibling sparse cache if it exists
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

    # Fallback: dense npz — stream rows from zip to avoid RAM spike
    print(f"    Dense SOAP cache: {cache_path}")
    print(f"    Streaming rows from zip to build sparse matrix ...")

    # Load cif_ids (small array, fine to load fully)
    # Reuse already-opened NpzFile handle from above.
    cif_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
               for x in data["cif_ids"]]
    # Resolve descriptor array key (older caches may use different names)
    descriptor_key = None
    for k in ("soap_descriptors", "descriptors", "soap_matrix", "X"):
        if k in data.files:
            descriptor_key = k
            break
    if descriptor_key is None:
        exclude = {"cif_ids", "species", "params", "sp_data", "sp_indices", "sp_indptr", "sp_shape"}
        candidates = [k for k in data.files if k not in exclude]
        if len(candidates) == 1:
            descriptor_key = candidates[0]
    if descriptor_key is None:
        raise KeyError(
            "Could not locate SOAP descriptor array inside cache. "
            f"Found keys: {data.files}"
        )
    print(f"    Descriptor key in cache: {descriptor_key}")

    # Get shape without loading the full array
    soap_shape = None
    import zipfile
    with zipfile.ZipFile(cache_path, 'r') as zf:
        with zf.open(f"{descriptor_key}.npy") as f:
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
    for i, row in enumerate(_read_npy_from_zip(cache_path, descriptor_key)):
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


def load_nominated_top_predictions(filepath):
    cids = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                cids.append(line)
    print(f"    Loaded {len(cids)} ensemble nominations")
    return cids


def load_bandgap_csv(csv_path):
    """Load DFT results CSV with flexible MOF ID column names."""
    results = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV appears empty or missing header: {csv_path}")
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
            results[name] = {"bandgap_eV": bg_val}
    print(f"    Loaded {len(results)} entries from bandgap CSV")
    return results


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
def panel_d_nominated(coords, is_labeled, bandgaps, threshold,
                   nominated_mask, nominated_cids, output_dir):
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
    n_p6 = int(nominated_mask.sum())
    if n_p6 > 0:
        ax.scatter(coords[nominated_mask, 0], coords[nominated_mask, 1],
                   c="#e41a1c", s=100, marker="*",
                   edgecolors="black", linewidths=0.5,
                   zorder=5, alpha=0.95)
        # Annotate names
        p6_coords = coords[nominated_mask]
        for k, cid in enumerate(nominated_cids):
            short = cid.replace("_FSR", "")
            if len(short) > 12:
                short = short[:10] + ".."
            ax.annotate(short, (p6_coords[k, 0], p6_coords[k, 1]),
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
    if n_p6 > 0:
        _leg.append(Line2D([], [], marker="*", color="w",
                           markerfacecolor="#e41a1c", markeredgecolor="black",
                           markeredgewidth=0.5, markersize=8,
                           label=f"Ensemble nominations ({n_p6})"))
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


def panel_e_nominated_hits(coords, is_labeled, hit_mask, hit_names, hit_bandgaps,
                           threshold, output_dir):
    """Extra panel: labeled/unlabeled background + nominated DFT hits only."""
    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    unlabeled_mask = ~is_labeled
    n_unlab = int(unlabeled_mask.sum())
    n_lab = int(is_labeled.sum())

    ax.scatter(coords[unlabeled_mask, 0], coords[unlabeled_mask, 1],
               c="#c0c0c0", s=1.6, alpha=0.30, rasterized=True, zorder=1)
    ax.scatter(coords[is_labeled, 0], coords[is_labeled, 1],
               c="#2171b5", s=2.1, alpha=0.42, rasterized=True, zorder=2)

    n_hits = int(hit_mask.sum())
    if n_hits > 0:
        hit_coords = coords[hit_mask]
        ax.scatter(hit_coords[:, 0], hit_coords[:, 1],
                   c="#f1c40f", s=165, marker="*",
                   edgecolors="black", linewidths=0.65,
                   zorder=6, alpha=0.98)
        for k in range(n_hits):
            short = hit_names[k].replace("_FSR", "")
            if len(short) > 24:
                short = short[:22] + ".."
            label = f"{short}\n{hit_bandgaps[k]:.3f} eV"
            ax.annotate(label, (hit_coords[k, 0], hit_coords[k, 1]),
                        fontsize=5.0, fontweight="bold",
                        xytext=(6, 6), textcoords="offset points",
                        color="#333333",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", ec="0.7", alpha=0.88, lw=0.3))

    _leg = [
        Line2D([], [], marker="o", color="w", markerfacecolor="#2171b5",
               markersize=5, label=f"DFT-labeled ({n_lab:,})"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#c0c0c0",
               markersize=5, label=f"Unlabeled ({n_unlab:,})"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#f1c40f",
               markeredgecolor="black", markeredgewidth=0.5, markersize=8,
               label=f"Nominated DFT hits < {threshold:.1f} eV ({n_hits})"),
    ]
    ax.legend(handles=_leg, loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.95, borderpad=0.5,
              handletextpad=0.5, handlelength=1.4, fontsize=7,
              markerscale=0.9)
    ax.set_title("(e)  Nominated DFT hits only  [SOAP space]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "soap_umap_e_nominated_hits")
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
                    help="Unlabeled-pool test_bandgaps_regression.json (unlabeled IDs)")
    pa.add_argument("--nominated_top_predictions", default=None,
                    help="File with top-K ensemble CIF IDs (one per line)")
    pa.add_argument("--bandgap_csv", default=None,
                    help="Optional DFT results CSV for nominations (for extra hits-only panel)")
    pa.add_argument("--soap_cache", default=None,
                    help="Pre-computed soap_descriptors.npz (skip SOAP stage)")
    pa.add_argument("--output_dir", default="./soap_umap_figures")
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
    nominated_cid_list = []
    if args.nominated_top_predictions and os.path.exists(args.nominated_top_predictions):
        nominated_cid_list = load_nominated_top_predictions(args.nominated_top_predictions)
    nom_dft = {}
    if args.bandgap_csv and os.path.exists(args.bandgap_csv):
        nom_dft = load_bandgap_csv(args.bandgap_csv)

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
    nominated_lookup = {c: True for c in nominated_cid_list}
    nominated_mask = np.array([_flex_lookup(c, nominated_lookup) is not None
                            for c in soap_cids])
    # For annotation: keep order aligned with mask
    nominated_present = [c for c in soap_cids
                      if _flex_lookup(c, nominated_lookup) is not None]

    n_lab = int(is_labeled.sum())
    n_ulab = n - n_lab
    print(f"\n    Final: {n} MOFs ({n_lab} labeled, {n_ulab} unlabeled)")
    if nominated_present:
        print(f"    Ensemble nominations in SOAP: {len(nominated_present)}/{len(nominated_cid_list)}")

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

    if nominated_present:
        panel_d_nominated(coords, is_labeled, bandgaps, args.threshold,
                       nominated_mask, nominated_present, args.output_dir)
    else:
        print("    Skipping panel (d) — no ensemble nominations provided")

    # Extra panel (e): nominations with DFT-confirmed hits only.
    n_hits_found = 0
    if nominated_present and nom_dft:
        nominated_lookup = {c: True for c in nominated_cid_list}
        hit_entries = []
        for cid in soap_cids:
            if _flex_lookup(cid, nominated_lookup) is None:
                continue
            info = _flex_lookup(cid, nom_dft)
            if info is None:
                continue
            bg = float(info.get("bandgap_eV", float("inf")))
            if np.isfinite(bg) and bg < args.threshold:
                hit_entries.append((cid, bg))
        hit_lookup = {cid: True for cid, _ in hit_entries}
        hit_mask = np.array([_flex_lookup(cid, hit_lookup) is not None for cid in soap_cids])
        hit_names = [cid for cid, _ in hit_entries]
        hit_bandgaps = [bg for _, bg in hit_entries]
        n_hits_found = len(hit_entries)
        panel_e_nominated_hits(coords, is_labeled, hit_mask, hit_names, hit_bandgaps,
                               args.threshold, args.output_dir)
    elif args.bandgap_csv:
        print("    Skipping panel (e) — no matched nominated hits from bandgap CSV")

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
        "ensemble_nominations_found": len(nominated_present),
        "nominated_dft_hits_found": n_hits_found,
    }
    sp = os.path.join(args.output_dir, "soap_umap_summary.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Done.  All outputs in {args.output_dir}/")
    print(f"  Panels: (a) labeled/unlabeled, (b) bandgap, (c) splits"
          + (", (d) ensemble nominations" if nominated_present else "")
          + (", (e) nominated DFT hits" if (nominated_present and nom_dft) else ""))
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
