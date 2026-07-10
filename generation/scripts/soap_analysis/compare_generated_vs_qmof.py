#!/usr/bin/env python3
"""
SOAP Comparison - generated-dataset vs QMOF
======================================

Computes per-MOF average SOAP descriptors for both the newly generated MOFs
(label = "generated-dataset") and QMOF (label = "QMOF"), then projects them into
a single 2D UMAP so we can see how well the generated points cover the QMOF cloud.

Data integrity (nothing is silently discarded)
-----------------------------------------------
- **Full SOAP** for each dataset is always stored under the canonical keys
  ``cif_ids``, ``soap_descriptors`` (float32, full dscribe layout),
  ``species``, ``params`` in the respective ``*_soap_descriptors.npz`` files.
  This script **never** overwrites those descriptor arrays with PCA-reduced
  vectors. If you pass ``--qmof-cache`` / ``--generated-cache``, those files on disk
  are **read only** (new SOAP is written only when computing from CIFs into
  ``--output_dir``).
- **PCA** (when used) is a *linear* preprocessing step applied **only** to
  build the matrix passed to UMAP. It is **lossy** in the strict linear-algebra
  sense, but the **original** vectors remain in the SOAP caches. For every run
  that uses PCA, we also write ``umap_soap_pca_projection.npz`` containing
  ``mean_``, ``components_``, and ``explained_variance_ratio_`` so anyone can
  **reconstruct the exact UMAP input** from the stacked full-SOAP matrix
  offline: ``(X - mean_) @ components_.T`` (same convention as scikit-learn's
  ``IncrementalPCA.transform``).
- **UMAP** maps high-(or medium-)dimensional points to 2D and is **always**
  lossy; that is intrinsic to the method, not a bug.

The SOAP hyperparameters, cache layout (.npz with keys cif_ids,
soap_descriptors, species, params), and the publication-style plotting
helpers are intentionally kept identical to figure_soap_analysis.py /
figure_soap_umap.py so that outputs look and load consistently with the
earlier QMOF work.

Two input modes per dataset
---------------------------
  --qmof-cif-dir / --generated-cif-dir
      Compute SOAP from CIFs using dscribe (periodic, average="inner").

  --qmof-cache / --generated-cache
      Reuse a .npz cache previously written by this script OR by
      figure_soap_analysis.py (keys cif_ids + soap_descriptors[/descriptors/
      embeddings]).  Skips the SOAP stage entirely.

Usage
-----
    python scripts/soap_analysis/compare_generated_vs_qmof.py \\
        --qmof-cif-dir <path to QMOF cifs> \\
        --generated-cif-dir  generated_cifs/small \\
        --output_dir   soap_analysis/generated_vs_qmof

Requirements:  dscribe, ase, numpy, matplotlib, umap-learn, scipy, tqdm,
                 scikit-learn (for IncrementalPCA when SOAP dim is large)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------- #
# SOAP parameters  (kept identical to figure_soap_analysis.py so the
# feature vectors from both scripts live in a comparable space)
# --------------------------------------------------------------------------- #
SOAP_RCUT = 6.0
SOAP_NMAX = 4
SOAP_LMAX = 4
SOAP_SIGMA = 0.5
SOAP_PERIODIC = True


# --------------------------------------------------------------------------- #
# Publication style + save helper - copied from figure_soap_umap.py so any
# figure this script emits looks indistinguishable from the earlier ones.
# --------------------------------------------------------------------------- #
def set_publication_style() -> None:
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


def _save_panel(fig, output_dir: Path, name: str) -> None:
    """Write the same figure as png + svg + pdf, matching the export format
    used by figure_soap_umap.py."""
    for fmt in ("png", "svg", "pdf"):
        p = output_dir / f"{name}.{fmt}"
        fig.savefig(p, dpi=600 if fmt == "png" else 300,
                    format=fmt, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    print(f"    Saved {name}.png / .svg / .pdf")


def _style_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=7, labelpad=2)
    ax.set_ylabel("UMAP-2", fontsize=7, labelpad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color("0.5")


# --------------------------------------------------------------------------- #
# SOAP computation - same structure as figure_soap_analysis.py
# (single "average=inner" vector per structure, species gathered up-front so
# every vector has the same layout, periodic=True for MOFs).
# --------------------------------------------------------------------------- #
def discover_cif_files(cif_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(cif_dir.glob("*.cif"))}


def compute_soap_descriptors(
    cif_dir: Path,
    cif_ids: list[str],
    output_path: Path,
    species_universe: list[str] | None = None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Compute periodic average-SOAP for each cif_id.

    If species_universe is provided the SOAP feature layout is fixed to that
    list (needed when comparing two datasets). Otherwise the species set is
    derived from the CIFs in this call.
    """
    from ase.io import read as ase_read
    from dscribe.descriptors import SOAP

    print("    Reading CIF files and collecting species ...")
    cif_map = discover_cif_files(cif_dir)
    structures: dict[str, object] = {}
    observed_species: set[str] = set()
    n_skip = 0
    skipped: list[str] = []

    t0 = time.time()
    for i, cid in enumerate(cif_ids):
        if cid not in cif_map:
            n_skip += 1
            if len(skipped) < 20:
                skipped.append(cid)
            continue
        try:
            atoms = ase_read(cif_map[cid])
            observed_species.update(atoms.get_chemical_symbols())
            structures[cid] = atoms
        except Exception as e:
            n_skip += 1
            if len(skipped) < 20:
                skipped.append(f"{cid}({e.__class__.__name__})")
        if (i + 1) % 2000 == 0:
            print(f"      Read {i+1}/{len(cif_ids)} CIFs ...")

    elapsed = time.time() - t0
    print(f"    Read {len(structures)} CIF files in {elapsed:.0f}s  (skipped {n_skip})")
    if skipped:
        print(f"    Skipped (sample): {skipped[:10]}")

    species = sorted(species_universe) if species_universe else sorted(observed_species)
    print(f"    SOAP species universe ({len(species)}): "
          f"{species[:20]}{'...' if len(species) > 20 else ''}")

    soap = SOAP(
        species=species,
        r_cut=SOAP_RCUT,
        n_max=SOAP_NMAX,
        l_max=SOAP_LMAX,
        sigma=SOAP_SIGMA,
        periodic=SOAP_PERIODIC,
        average="inner",
        sparse=False,
    )
    soap_dim = soap.get_number_of_features()
    print(f"    SOAP descriptor dimension: {soap_dim}")

    print("    Computing SOAP descriptors ...")
    # Preallocate the output float32 matrix so we never hold both a Python list
    # of per-MOF float64 arrays AND the final stacked array at the same time.
    # That list-then-stack pattern is what pushes peak RAM over the edge when
    # one dataset already occupies ~20 GB (QMOF cache) and the other is ~13 GB
    # in flight - the combined peak was ~60 GB on a 64 GB node.
    total_structs = len(structures)
    soap_matrix = np.empty((total_structs, soap_dim), dtype=np.float32)
    ordered_ids: list[str] = []
    row_idx = 0
    t0 = time.time()
    last_report = t0

    for cid in cif_ids:
        if cid not in structures:
            continue
        try:
            desc = soap.create(structures[cid])
            if desc.ndim == 2:
                desc = desc[0]
            # astype(..., copy=False) avoids an extra copy when dscribe already
            # returns float32; otherwise this is a single float64->float32 cast
            # the size of one MOF (~1 MB), not the whole matrix.
            soap_matrix[row_idx] = np.asarray(desc, dtype=np.float32)
            ordered_ids.append(cid)
            row_idx += 1
        except Exception:
            continue

        now = time.time()
        if now - last_report >= 30:
            last_report = now
            done = row_idx
            total = total_structs
            rate = done / (now - t0) if (now - t0) > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(remaining), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            print(f"      [{done:>6}/{total}]  {done/total*100:5.1f}%  "
                  f"ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s  "
                  f"({rate:.1f} MOF/s)", flush=True)

    # Trim to the number of rows that were actually written (some MOFs may
    # have raised inside soap.create and been skipped).
    if row_idx < total_structs:
        soap_matrix = soap_matrix[:row_idx].copy()
    if row_idx == 0:
        soap_matrix = np.empty((0, soap_dim), dtype=np.float32)
    el = time.time() - t0
    em, es = divmod(int(el), 60)
    eh, em = divmod(em, 60)
    print(f"    SOAP done: {len(ordered_ids)} MOFs, dim={soap_matrix.shape[1]}  "
          f"[{eh}h{em:02d}m{es:02d}s]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        cif_ids=np.array(ordered_ids),
        soap_descriptors=soap_matrix,
        species=np.array(species),
        params=np.array([SOAP_RCUT, SOAP_NMAX, SOAP_LMAX, SOAP_SIGMA]),
    )
    print(f"    SOAP cache saved -> {output_path}")
    return ordered_ids, soap_matrix, species


def _save_pca_projection_artifacts(
    output_dir: Path,
    ipca: object,
    soap_feature_dim: int,
    n_rows: int,
    pca_batch: int,
) -> None:
    """Persist IncrementalPCA state so the UMAP input can be reproduced from
    full SOAP without re-fitting. ``mean_`` / ``components_`` are float32;
    ``explained_variance_ratio_`` stays float64 for summation accuracy."""
    mean_ = np.asarray(ipca.mean_, dtype=np.float32)
    comp = np.asarray(ipca.components_, dtype=np.float32)
    evr = np.asarray(ipca.explained_variance_ratio_, dtype=np.float64)
    out_npz = output_dir / "umap_soap_pca_projection.npz"
    np.savez_compressed(
        out_npz,
        mean_=mean_,
        components_=comp,
        explained_variance_ratio_=evr,
    )
    meta = {
        "soap_feature_dim": int(soap_feature_dim),
        "n_components": int(comp.shape[0]),
        "n_samples_seen": int(getattr(ipca, "n_samples_seen_", n_rows)),
        "pca_batch_effective": int(pca_batch),
        "reconstruct_umap_input": (
            "Let X be (n, soap_feature_dim) float32, same row order as "
            "soap_comparison_row_manifest.npz / stacked caches. "
            "UMAP input = (X - mean_) @ components_.T  "
            "(scikit-learn IncrementalPCA.transform convention)."
        ),
        "projection_arrays": {
            "mean_": "float32 (n_features,)",
            "components_": "float32 (n_components, n_features)",
            "explained_variance_ratio_": "float64 (n_components,)",
        },
    }
    meta_path = output_dir / "umap_soap_pca_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"    PCA projection + meta saved -> {out_npz} , {meta_path}")


def _save_row_manifest(output_dir: Path, cif_ids: np.ndarray, labels: np.ndarray) -> None:
    """Lightweight audit trail: row order matches stacked SOAP / UMAP."""
    out = output_dir / "soap_comparison_row_manifest.npz"
    np.savez_compressed(
        out,
        cif_ids=cif_ids.astype(object),
        labels=labels.astype(object),
    )
    print(f"    Row manifest (cif_id + label per UMAP row) -> {out}")


def load_soap_cache(cache_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Load a dense SOAP cache. Accepts the canonical key ``soap_descriptors``
    from figure_soap_analysis.py as well as legacy aliases (``descriptors``,
    ``embeddings``, ``soap_matrix``, ``X``)."""
    data = np.load(cache_path, allow_pickle=True)
    cif_ids = [str(c) for c in data["cif_ids"]]
    key = None
    for k in ("soap_descriptors", "descriptors", "embeddings", "soap_matrix", "X"):
        if k in data.files:
            key = k
            break
    if key is None:
        raise KeyError(f"Could not locate SOAP descriptor array in {cache_path}; "
                       f"found keys: {data.files}")
    soap_matrix = np.asarray(data[key], dtype=np.float32)
    species = [str(s) for s in data["species"]] if "species" in data.files else []
    params = data["params"] if "params" in data.files else None
    print(f"    Loaded SOAP cache: {cache_path}")
    print(f"    {len(cif_ids)} MOFs, dim={soap_matrix.shape[1]}, key='{key}'")
    if params is not None:
        print(f"    Cache params: r_cut={params[0]}, n_max={int(params[1])}, "
              f"l_max={int(params[2])}, sigma={params[3]}")
    return cif_ids, soap_matrix, species


# --------------------------------------------------------------------------- #
# UMAP
# --------------------------------------------------------------------------- #
def compute_umap(matrix: np.ndarray, n_neighbors: int, min_dist: float,
                 seed: int, n_jobs_umap: int | None = None) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError:
        sys.exit("ERROR: umap-learn not installed. pip install umap-learn")
    n, d = matrix.shape
    print(f"    {n} points, dim={d} ...")
    # n_jobs=-1 duplicates index data across workers and can OOM on large d;
    # single-thread is slower but much safer for 10k+ points × k dims.
    if n_jobs_umap is None:
        n_jobs_umap = 1 if d > 2000 else -1
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        n_jobs=n_jobs_umap,
        low_memory=True,
    )
    coords = reducer.fit_transform(matrix)
    print(f"    UMAP done -> {coords.shape}")
    return coords


# --------------------------------------------------------------------------- #
# Overlay panel - two datasets in one UMAP (QMOF background + generated overlay)
# --------------------------------------------------------------------------- #
def panel_dataset_overlay(
    coords: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.3))

    qmask = labels == "QMOF"
    emask = labels == "generated-dataset"
    n_qmof = int(qmask.sum())
    n_generated = int(emask.sum())

    # QMOF drawn first as diffuse background, generated overlaid on top so we can
    # visually read coverage at a glance.
    ax.scatter(coords[qmask, 0], coords[qmask, 1],
               c="#2171b5", s=2.0, alpha=0.35, rasterized=True, zorder=1)
    ax.scatter(coords[emask, 0], coords[emask, 1],
               c="#e41a1c", s=2.2, alpha=0.55, rasterized=True, zorder=2)

    legend = [
        Line2D([], [], marker="o", color="w", markerfacecolor="#2171b5",
               markersize=5, label=f"QMOF ({n_qmof:,})"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#e41a1c",
               markersize=5, label=f"generated-dataset ({n_generated:,})"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.95, borderpad=0.5,
              handletextpad=0.5, handlelength=1.4, fontsize=7)
    ax.set_title("SOAP UMAP: generated-dataset vs. QMOF",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "soap_umap_generated_vs_qmof")
    plt.close()


def panel_density_contour(
    coords: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    """Side-by-side: same UMAP grid, but showing each dataset's 2D density
    with KDE contours so overlap regions are visually obvious."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("    scipy not available - skipping density panel.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.3),
                                   sharex=True, sharey=True)
    qmask = labels == "QMOF"
    emask = labels == "generated-dataset"

    for ax, mask, title, color in [
        (ax1, qmask, f"QMOF ({int(qmask.sum()):,})", "#2171b5"),
        (ax2, emask, f"generated-dataset ({int(emask.sum()):,})", "#e41a1c"),
    ]:
        # Faint gray background of all points for context
        ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=1.2,
                   alpha=0.35, rasterized=True, zorder=1)
        sub = coords[mask]
        if sub.shape[0] >= 20:
            kde = gaussian_kde(sub.T)
            xg = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 120)
            yg = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 120)
            X, Y = np.meshgrid(xg, yg)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=6, colors=color, linewidths=0.6, zorder=3)
        ax.scatter(sub[:, 0], sub[:, 1], c=color, s=1.6, alpha=0.55,
                   rasterized=True, zorder=2)
        ax.set_title(title, fontweight="bold", fontsize=9)
        _style_ax(ax)

    fig.suptitle("SOAP UMAP density - where each dataset concentrates",
                 fontweight="bold", fontsize=9, y=1.01)
    plt.tight_layout()
    _save_panel(fig, output_dir, "soap_umap_density_generated_vs_qmof")
    plt.close()


# --------------------------------------------------------------------------- #
# Pipeline per dataset
# --------------------------------------------------------------------------- #
def prepare_dataset(
    label: str,
    cif_dir: Path | None,
    cache_path: Path | None,
    limit: int,
    species_universe: list[str],
    output_cache: Path,
) -> tuple[list[str], np.ndarray]:
    """Return (cif_ids, soap_matrix) for one dataset, either by loading a
    cache or by computing from CIFs using the shared species universe."""
    if cache_path is not None and cache_path.exists():
        cif_ids, soap_matrix, _ = load_soap_cache(cache_path)
        if limit > 0 and len(cif_ids) > limit:
            cif_ids = cif_ids[:limit]
            soap_matrix = soap_matrix[:limit]
            print(f"    [{label}] capped to {limit} cached rows")
        return cif_ids, soap_matrix

    assert cif_dir is not None, f"[{label}] need --{label}-cif-dir or --{label}-cache"
    cif_ids_full = list(discover_cif_files(cif_dir).keys())
    if limit > 0:
        cif_ids_full = cif_ids_full[:limit]
    cif_ids, soap_matrix, _ = compute_soap_descriptors(
        cif_dir, cif_ids_full, output_cache,
        species_universe=species_universe,
    )
    return cif_ids, soap_matrix


def build_species_universe(
    qmof_cif_dir: Path | None,
    generated_cif_dir: Path | None,
    qmof_cache: Path | None,
    generated_cache: Path | None,
) -> list[str]:
    """Determine the species list shared by both SOAP computations. If a
    cache is used for one side we inherit its species list; if both sides
    compute fresh, we scan the union of their CIFs."""
    species: set[str] = set()

    # Inherit from whichever cache(s) we have
    for cache in (qmof_cache, generated_cache):
        if cache is not None and cache.exists():
            data = np.load(cache, allow_pickle=True)
            if "species" in data.files:
                species.update(str(s) for s in data["species"])

    if species:
        print(f"    Species universe inherited from cache(s): "
              f"{sorted(species)[:20]}{'...' if len(species) > 20 else ''}")
        return sorted(species)

    from ase.io import read as ase_read
    dirs = [d for d in (qmof_cif_dir, generated_cif_dir) if d is not None]
    print(f"    Scanning {len(dirs)} CIF dir(s) to build shared species universe")
    for d in dirs:
        for p in sorted(d.glob("*.cif")):
            try:
                atoms = ase_read(p)
                species.update(atoms.get_chemical_symbols())
            except Exception:
                continue
    print(f"    Species universe ({len(species)}): "
          f"{sorted(species)[:20]}{'...' if len(species) > 20 else ''}")
    return sorted(species)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    pa = argparse.ArgumentParser(
        description="SOAP comparison of generated generated-dataset against QMOF",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    pa.add_argument("--qmof-cif-dir", type=Path, default=None,
                    help="Directory of QMOF .cif files (ignored if --qmof-cache is set).")
    pa.add_argument("--generated-cif-dir", type=Path, default=None,
                    help="Directory of generated MOF .cif files (the generated-dataset).")
    pa.add_argument("--qmof-cache", type=Path, default=None,
                    help="Cached soap_descriptors.npz for QMOF "
                         "(keys: cif_ids + {soap_descriptors|descriptors|embeddings}).")
    pa.add_argument("--generated-cache", type=Path, default=None,
                    help="Cached soap_descriptors.npz for generated-dataset.")
    pa.add_argument("--max-qmof", type=int, default=0,
                    help="Cap QMOF sample size (0 = all).")
    pa.add_argument("--max-generated", type=int, default=0,
                    help="Cap generated-dataset sample size (0 = all).")
    pa.add_argument("--output_dir", type=Path,
                    default=Path("soap_analysis/generated_vs_qmof"))
    pa.add_argument("--n_neighbors", type=int, default=30)
    pa.add_argument("--min_dist", type=float, default=0.3)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--save_umap_cache", action="store_true",
                    help="Persist UMAP coordinates for fast re-plotting.")
    pa.add_argument("--pca-dim", type=int, default=0,
                    help="If > 0, reduce SOAP to exactly this many PCA "
                         "components before UMAP (overrides --pca-auto-dim). "
                         "If 0, auto mode may apply (see --pca-auto-dim).")
    pa.add_argument(
        "--pca-auto-dim",
        type=int,
        default=512,
        help="When SOAP dim > 12000 and --pca-dim is 0 and --full-soap-umap is "
             "not set, use this many PCA components (capped at n_MOFS-1). "
             "Higher retains more variance; 512 is a strong default for "
             "analysis quality on 64G nodes.",
    )
    pa.add_argument("--full-soap-umap", action="store_true",
                    help="Run UMAP on full SOAP vectors (no PCA). Expect "
                         "OOM on typical 64G nodes when dim ~2e5.")
    pa.add_argument("--pca-batch", type=int, default=512,
                    help="Batch size for IncrementalPCA (only used when "
                         "PCA runs; raised automatically if smaller than "
                         "n_components).")
    pa.add_argument(
        "--no-save-pca-state",
        action="store_true",
        help="Skip writing umap_soap_pca_projection.npz and "
             "umap_soap_pca_meta.json (not recommended; you lose exact "
             "reproducibility of the UMAP input from full SOAP).",
    )
    args = pa.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    print("=" * 70)
    print("  SOAP COMPARISON - generated-dataset vs QMOF")
    print("=" * 70)

    if args.qmof_cache is None and args.qmof_cif_dir is None:
        pa.error("Need --qmof-cache or --qmof-cif-dir")
    if args.generated_cache is None and args.generated_cif_dir is None:
        pa.error("Need --generated-cache or --generated-cif-dir")

    # Share one species universe across both SOAP calculations so the
    # feature vectors are comparable row-for-row.
    print("\n[1/3] Species universe ...")
    need_compute = (args.qmof_cache is None) or (args.generated_cache is None)
    if need_compute:
        species_universe = build_species_universe(
            args.qmof_cif_dir, args.generated_cif_dir,
            args.qmof_cache, args.generated_cache,
        )
    else:
        species_universe = []  # both sides cached, trust their layouts

    print("\n[2/3] Computing / loading SOAP ...")
    qmof_ids, qmof_mat = prepare_dataset(
        "qmof",
        args.qmof_cif_dir, args.qmof_cache, args.max_qmof,
        species_universe,
        args.output_dir / "qmof_soap_descriptors.npz",
    )
    generated_ids, generated_mat = prepare_dataset(
        "generated",
        args.generated_cif_dir, args.generated_cache, args.max_generated,
        species_universe,
        args.output_dir / "generated_soap_descriptors.npz",
    )

    if qmof_mat.size == 0 or generated_mat.size == 0:
        sys.exit("One of the SOAP matrices is empty - aborting.")
    if qmof_mat.shape[1] != generated_mat.shape[1]:
        sys.exit(f"SOAP dim mismatch: QMOF={qmof_mat.shape[1]} vs "
                 f"generated={generated_mat.shape[1]}. Recompute both with the same "
                 "species universe (delete the offending cache).")
    if qmof_mat.shape[0] != len(qmof_ids) or generated_mat.shape[0] != len(generated_ids):
        sys.exit(
            "SOAP matrix row count does not match cif_ids length "
            f"(QMOF rows={qmof_mat.shape[0]} vs ids={len(qmof_ids)}; "
            f"generated rows={generated_mat.shape[0]} vs ids={len(generated_ids)})."
        )

    soap_feature_dim = int(qmof_mat.shape[1])
    n_total = len(qmof_ids) + len(generated_ids)
    print(f"\n    Combined: {len(qmof_ids)} QMOF + {len(generated_ids)} generated  "
          f"(SOAP dim = {soap_feature_dim}, n_total={n_total})")

    print("\n[3/3] Computing joint UMAP and plotting ...")
    # Build the combined matrix directly into a fresh buffer and then release
    # the per-dataset matrices.  np.vstack would briefly hold three copies
    # (qmof_mat + generated_mat + combined); filling a preallocated buffer and
    # del-ing the sources keeps peak RAM at roughly one combined matrix.
    import gc
    n_q, n_e = len(qmof_ids), len(generated_ids)
    feat_dim = soap_feature_dim
    combined = np.empty((n_q + n_e, feat_dim), dtype=np.float32)
    combined[:n_q] = qmof_mat
    combined[n_q:] = generated_mat
    del qmof_mat, generated_mat
    gc.collect()

    labels = np.array(["QMOF"] * n_q + ["generated-dataset"] * n_e)
    cif_ids = np.array(qmof_ids + generated_ids)
    _save_row_manifest(args.output_dir, cif_ids, labels)

    # UMAP on (N ~ 34k) × (SOAP dim ~ 2.4e5) float32 blows past 64G RAM even
    # when caches load fine — the NN graph + index copies dominate.  Full SOAP
    # vectors remain on disk in the .npz caches; PCA here is only for the 2D
    # embedding step (standard practice for high-d descriptors).
    pca_dim_effective = int(args.pca_dim)
    pca_explained_sum: float | None = None
    pca_k_applied: int | None = None
    pca_evr_per_component: list[float] | None = None
    pca_dim_target: int | None = None
    pca_auto_triggered = False
    if pca_dim_effective <= 0 and feat_dim > 12000 and not args.full_soap_umap:
        pca_auto_triggered = True
        cap = min(int(args.pca_auto_dim), n_total - 1)
        pca_dim_effective = max(cap, 1)
        print(
            f"    Auto: SOAP dim > 12000 -> IncrementalPCA({pca_dim_effective}) "
            "before UMAP (64G-safe). Full SOAP remains in the per-dataset "
            "*.npz caches unchanged; PCA state is written for exact reproduction "
            "of the UMAP input. Use --full-soap-umap for raw-SOAP UMAP "
            "(~128G+ RAM typical)."
        )
    if pca_dim_effective > 0:
        try:
            from sklearn.decomposition import IncrementalPCA
        except ImportError:
            sys.exit(
                "ERROR: scikit-learn required for PCA before UMAP "
                "(pip install scikit-learn)."
            )
        n = combined.shape[0]
        k = min(pca_dim_effective, n - 1)
        if k < 1:
            sys.exit("Not enough rows for PCA (need at least 2 MOFs).")
        # batch must be >= n_components for IncrementalPCA and <= n for the
        # last partial_fit slice on small N.
        batch = min(max(int(args.pca_batch), k + 1), n)
        print(f"    IncrementalPCA: {combined.shape[1]} -> {k} dims "
              f"(batch={batch}) ...")
        pca_dim_target = int(pca_dim_effective)
        ipca = IncrementalPCA(n_components=k, batch_size=batch)
        for start in range(0, n, batch):
            ipca.partial_fit(combined[start:start + batch])
        reduced = np.empty((n, k), dtype=np.float32)
        for start in range(0, n, batch):
            reduced[start:start + batch] = ipca.transform(
                combined[start:start + batch]
            ).astype(np.float32, copy=False)
        evr = np.asarray(ipca.explained_variance_ratio_, dtype=np.float64)
        pca_evr_per_component = [float(x) for x in evr.tolist()]
        pca_explained_sum = float(np.sum(evr))
        pca_k_applied = k
        if not args.no_save_pca_state:
            _save_pca_projection_artifacts(
                args.output_dir, ipca, soap_feature_dim, n, batch,
            )
        else:
            print("    (--no-save-pca-state: skipping PCA projection dump)")
        del combined
        gc.collect()
        combined = reduced
        print(f"    PCA done -> {combined.shape}, "
              f"explained variance ratio sum = {pca_explained_sum:.4f}")

    umap_input_dim = int(combined.shape[1])
    coords = compute_umap(combined, args.n_neighbors, args.min_dist, args.seed)

    panel_dataset_overlay(coords, labels, args.output_dir)
    panel_density_contour(coords, labels, args.output_dir)

    if args.save_umap_cache:
        cp = args.output_dir / "soap_umap_cache.npz"
        np.savez_compressed(cp, coords=coords, labels=labels, cif_ids=cif_ids)
        print(f"    UMAP cache saved -> {cp}")

    pca_proj_path = args.output_dir / "umap_soap_pca_projection.npz"

    summary = {
        "total_mofs": int(len(qmof_ids) + len(generated_ids)),
        "n_qmof": int(len(qmof_ids)),
        "n_generated": int(len(generated_ids)),
        "soap_dim": soap_feature_dim,
        "umap_input_dim": umap_input_dim,
        "pca_before_umap": pca_k_applied is not None,
        "pca_auto_triggered": pca_auto_triggered,
        "pca_auto_dim_setting": int(args.pca_auto_dim),
        "pca_dim_target": pca_dim_target,
        "pca_dim_applied": pca_k_applied,
        "pca_dim_clipped": (
            (pca_dim_target is not None and pca_k_applied is not None
             and pca_dim_target > pca_k_applied)
        ),
        "pca_explained_variance_ratio_sum": pca_explained_sum,
        "pca_explained_variance_ratio": pca_evr_per_component,
        "pca_cumulative_explained_variance_ratio": (
            [float(x) for x in np.cumsum(pca_evr_per_component).tolist()]
            if pca_evr_per_component else None
        ),
        "full_soap_umap": bool(args.full_soap_umap),
        "no_save_pca_state": bool(args.no_save_pca_state),
        "artifacts": {
            "row_manifest": str(
                (args.output_dir / "soap_comparison_row_manifest.npz").resolve()
            ),
            "pca_projection": str(pca_proj_path.resolve())
            if pca_proj_path.is_file() else None,
            "pca_meta_json": str(
                (args.output_dir / "umap_soap_pca_meta.json").resolve()
            ) if (args.output_dir / "umap_soap_pca_meta.json").is_file() else None,
        },
        "inputs": {
            "qmof_cache": str(args.qmof_cache.resolve())
            if args.qmof_cache and args.qmof_cache.is_file() else None,
            "generated_cache": str(args.generated_cache.resolve())
            if args.generated_cache and args.generated_cache.is_file() else None,
            "qmof_cif_dir": str(args.qmof_cif_dir.resolve())
            if args.qmof_cif_dir else None,
            "generated_cif_dir": str(args.generated_cif_dir.resolve())
            if args.generated_cif_dir else None,
        },
        "soap_params": {
            "r_cut": SOAP_RCUT, "n_max": SOAP_NMAX,
            "l_max": SOAP_LMAX, "sigma": SOAP_SIGMA,
            "periodic": SOAP_PERIODIC, "average": "inner",
        },
        "umap_params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
            "seed": args.seed,
        },
    }
    with (args.output_dir / "soap_comparison_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Done. All outputs in {args.output_dir}/")
    print(f"  Panels: soap_umap_generated_vs_qmof.(png|svg|pdf), "
          f"soap_umap_density_generated_vs_qmof.(png|svg|pdf)")
    print("  Audit: soap_comparison_summary.json , "
          "soap_comparison_row_manifest.npz")
    if pca_k_applied is not None and not args.no_save_pca_state:
        print("  PCA (reproduce UMAP input from full SOAP): "
              "umap_soap_pca_projection.npz , umap_soap_pca_meta.json")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
