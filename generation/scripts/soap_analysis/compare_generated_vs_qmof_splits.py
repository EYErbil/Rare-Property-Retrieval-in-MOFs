#!/usr/bin/env python3
"""
SOAP UMAP: Generated MOFs vs QMOF train/val/test splits
=======================================================

Creates one joint SOAP space where:
- QMOF points are restricted to split-labeled IDs from
  {train,val,test}_bandgaps_regression.json
- Generated MOFs use all available IDs
- UMAP colors are split-specific for QMOF plus one color for generated

This script is a new entrypoint and does not modify existing SOAP scripts.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SOAP_RCUT = 6.0
SOAP_NMAX = 4
SOAP_LMAX = 4
SOAP_SIGMA = 0.5
SOAP_PERIODIC = True


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
    })


def _save_panel(fig, output_dir: Path, name: str) -> None:
    for fmt in ("png", "svg", "pdf"):
        p = output_dir / f"{name}.{fmt}"
        fig.savefig(
            p,
            dpi=600 if fmt == "png" else 300,
            format=fmt,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    print(f"    Saved {name}.png / .svg / .pdf")


def _style_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=7, labelpad=2)
    ax.set_ylabel("UMAP-2", fontsize=7, labelpad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color("0.5")


def discover_cif_files(cif_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(cif_dir.glob("*.cif"))}


def _id_variants(cid: str):
    yield cid
    bare = cid.replace(".cif", "")
    if bare != cid:
        yield bare
    if "_FSR" in bare:
        yield bare.replace("_FSR", "")
    else:
        yield bare + "_FSR"


def _flex_lookup(cid: str, lookup_dict: dict[str, str]):
    for v in _id_variants(cid):
        if v in lookup_dict:
            return lookup_dict[v]
    return None


def load_split_assignments(splits_dir: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    found = 0
    for split_name in ("train", "val", "test"):
        p = splits_dir / f"{split_name}_bandgaps_regression.json"
        if not p.exists():
            print(f"    Missing split file: {p}")
            continue
        with p.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        for cid in d.keys():
            assignments[str(cid)] = split_name
        found += 1
        print(f"    {split_name}: {len(d)} IDs")
    if found == 0:
        sys.exit(f"No split files found in {splits_dir}")
    return assignments


def compute_soap_descriptors(
    cif_dir: Path,
    cif_ids: list[str],
    output_path: Path,
    species_universe: list[str] | None = None,
) -> tuple[list[str], np.ndarray, list[str]]:
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
    print(
        f"    SOAP species universe ({len(species)}): "
        f"{species[:20]}{'...' if len(species) > 20 else ''}"
    )

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
            print(
                f"      [{done:>6}/{total}]  {done/total*100:5.1f}%  "
                f"ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s  "
                f"({rate:.1f} MOF/s)",
                flush=True,
            )

    if row_idx < total_structs:
        soap_matrix = soap_matrix[:row_idx].copy()
    if row_idx == 0:
        soap_matrix = np.empty((0, soap_dim), dtype=np.float32)

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


def load_soap_cache(cache_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    data = np.load(cache_path, allow_pickle=True)
    cif_ids = [str(c) for c in data["cif_ids"]]
    key = None
    for k in ("soap_descriptors", "descriptors", "embeddings", "soap_matrix", "X"):
        if k in data.files:
            key = k
            break
    if key is None:
        raise KeyError(f"No SOAP descriptor key found in {cache_path}; keys={data.files}")
    soap_matrix = np.asarray(data[key], dtype=np.float32)
    species = [str(s) for s in data["species"]] if "species" in data.files else []
    print(f"    Loaded SOAP cache: {cache_path}")
    print(f"    {len(cif_ids)} rows, dim={soap_matrix.shape[1]}, key='{key}'")
    return cif_ids, soap_matrix, species


def build_species_universe(
    qmof_cif_dir: Path | None,
    generated_cif_dir: Path | None,
    qmof_cache: Path | None,
    generated_cache: Path | None,
) -> list[str]:
    species: set[str] = set()
    for cache in (qmof_cache, generated_cache):
        if cache is not None and cache.exists():
            data = np.load(cache, allow_pickle=True)
            if "species" in data.files:
                species.update(str(s) for s in data["species"])
    if species:
        print(
            f"    Species universe inherited from cache(s): "
            f"{sorted(species)[:20]}{'...' if len(species) > 20 else ''}"
        )
        return sorted(species)

    from ase.io import read as ase_read

    dirs = [d for d in (qmof_cif_dir, generated_cif_dir) if d is not None]
    print(f"    Scanning {len(dirs)} CIF dir(s) for shared species universe")
    for d in dirs:
        for p in sorted(d.glob("*.cif")):
            try:
                atoms = ase_read(p)
                species.update(atoms.get_chemical_symbols())
            except Exception:
                continue
    return sorted(species)


def subset_qmof_to_splits(
    qmof_ids: list[str],
    qmof_mat: np.ndarray,
    split_assignments: dict[str, str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    keep_idx: list[int] = []
    split_labels: list[str] = []
    for i, cid in enumerate(qmof_ids):
        sp = _flex_lookup(cid, split_assignments)
        if sp in ("train", "val", "test"):
            keep_idx.append(i)
            split_labels.append(sp)
    if not keep_idx:
        sys.exit("No QMOF rows matched train/val/test split assignments.")
    idx = np.asarray(keep_idx, dtype=np.int64)
    return [qmof_ids[i] for i in keep_idx], qmof_mat[idx], np.asarray(split_labels, dtype=object)


def compute_umap(
    matrix: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError:
        sys.exit("ERROR: umap-learn not installed. pip install umap-learn")

    n, d = matrix.shape
    print(f"    UMAP input: {n} points, dim={d}")
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        n_jobs=1 if d > 2000 else -1,
        low_memory=True,
    )
    coords = reducer.fit_transform(matrix)
    print(f"    UMAP done -> {coords.shape}")
    return coords


def panel_overlay(coords: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    color_map = {
        "generated": "#e41a1c",
        "train": "#4292c6",
        "val": "#fd8d3c",
        "test": "#969696",
    }

    draw_order = ("generated", "test", "val", "train")
    for name in draw_order:
        m = labels == name
        if not np.any(m):
            continue
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            c=color_map[name],
            s=2.2 if name == "generated" else 2.0,
            alpha=0.50 if name == "generated" else 0.45,
            rasterized=True,
            zorder=1 if name == "generated" else 2,
        )

    legend = []
    for name in ("generated", "train", "val", "test"):
        count = int(np.sum(labels == name))
        if count == 0:
            continue
        legend.append(
            Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=color_map[name],
                markersize=5,
                label=f"{name.capitalize()} ({count:,})",
            )
        )
    ax.legend(
        handles=legend,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        framealpha=0.95,
        borderpad=0.5,
        handletextpad=0.5,
        handlelength=1.4,
        fontsize=7,
    )
    ax.set_title("SOAP UMAP: Generated vs QMOF train/val/test", fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "soap_umap_generated_vs_qmof_splits")
    plt.close()


def main() -> int:
    pa = argparse.ArgumentParser(
        description="Compare generated MOFs to QMOF train/val/test in SOAP-UMAP space."
    )
    pa.add_argument("--qmof-cif-dir", type=Path, default=None)
    pa.add_argument("--generated-cif-dir", type=Path, default=None)
    pa.add_argument("--qmof-cache", type=Path, default=None)
    pa.add_argument("--generated-cache", type=Path, default=None)
    pa.add_argument("--labeled-splits-dir", type=Path, required=True)
    pa.add_argument("--max-qmof", type=int, default=0)
    pa.add_argument("--max-generated", type=int, default=0)
    pa.add_argument("--output_dir", type=Path, default=Path("soap_analysis/generated_vs_qmof_splits"))
    pa.add_argument("--n_neighbors", type=int, default=30)
    pa.add_argument("--min_dist", type=float, default=0.3)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--pca-dim", type=int, default=512, help="Set 0 to disable PCA before UMAP.")
    pa.add_argument("--pca-batch", type=int, default=512)
    args = pa.parse_args()

    if args.qmof_cache is None and args.qmof_cif_dir is None:
        pa.error("Need --qmof-cache or --qmof-cif-dir")
    if args.generated_cache is None and args.generated_cif_dir is None:
        pa.error("Need --generated-cache or --generated-cif-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    print("=" * 70)
    print("  SOAP COMPARISON - Generated vs QMOF splits")
    print("=" * 70)

    print("\n[1/4] Loading split assignments ...")
    split_assignments = load_split_assignments(args.labeled_splits_dir)
    split_id_count = len(split_assignments)
    print(f"    Total split-labeled IDs: {split_id_count}")

    print("\n[2/4] Computing / loading SOAP ...")
    species_universe = build_species_universe(
        args.qmof_cif_dir,
        args.generated_cif_dir,
        args.qmof_cache,
        args.generated_cache,
    )

    if args.qmof_cache is not None and args.qmof_cache.exists():
        qmof_ids, qmof_mat, _ = load_soap_cache(args.qmof_cache)
    else:
        assert args.qmof_cif_dir is not None
        qmof_all_ids = list(discover_cif_files(args.qmof_cif_dir).keys())
        if args.max_qmof > 0:
            qmof_all_ids = qmof_all_ids[: args.max_qmof]
        qmof_ids, qmof_mat, _ = compute_soap_descriptors(
            args.qmof_cif_dir,
            qmof_all_ids,
            args.output_dir / "qmof_soap_descriptors.npz",
            species_universe=species_universe,
        )

    qmof_ids_split, qmof_mat_split, qmof_split_labels = subset_qmof_to_splits(
        qmof_ids, qmof_mat, split_assignments
    )
    n_unmatched_qmof = len(qmof_ids) - len(qmof_ids_split)
    print(f"    QMOF rows kept in train/val/test: {len(qmof_ids_split)}")
    print(f"    QMOF rows excluded (not in split files): {n_unmatched_qmof}")

    # Critical for memory: release full QMOF cache before loading generated cache.
    # The full QMOF matrix can be ~20 GB; keeping both full matrices at once
    # often triggers OOM on 64 GB nodes.
    del qmof_mat, qmof_ids
    gc.collect()

    if args.generated_cache is not None and args.generated_cache.exists():
        generated_ids, generated_mat, _ = load_soap_cache(args.generated_cache)
    else:
        assert args.generated_cif_dir is not None
        generated_all_ids = list(discover_cif_files(args.generated_cif_dir).keys())
        if args.max_generated > 0:
            generated_all_ids = generated_all_ids[: args.max_generated]
        generated_ids, generated_mat, _ = compute_soap_descriptors(
            args.generated_cif_dir,
            generated_all_ids,
            args.output_dir / "generated_soap_descriptors.npz",
            species_universe=species_universe,
        )

    if qmof_mat_split.shape[1] != generated_mat.shape[1]:
        sys.exit(
            f"SOAP dim mismatch: QMOF={qmof_mat_split.shape[1]} vs generated={generated_mat.shape[1]}"
        )

    print("\n[3/4] Joint UMAP ...")
    n_q = len(qmof_ids_split)
    n_e = len(generated_ids)
    feat_dim = qmof_mat_split.shape[1]
    combined = np.empty((n_q + n_e, feat_dim), dtype=np.float32)
    combined[:n_q] = qmof_mat_split
    combined[n_q:] = generated_mat
    del qmof_mat_split, generated_mat
    gc.collect()
    labels = np.array(list(qmof_split_labels) + (["generated"] * n_e), dtype=object)
    all_ids = np.array(qmof_ids_split + generated_ids, dtype=object)

    pca_dim_applied = None
    pca_explained = None
    if args.pca_dim > 0:
        try:
            from sklearn.decomposition import IncrementalPCA
        except ImportError:
            sys.exit("ERROR: scikit-learn required for --pca-dim > 0")

        n = combined.shape[0]
        k = min(args.pca_dim, n - 1)
        if k < 1:
            sys.exit("Not enough rows for PCA.")
        batch = min(max(int(args.pca_batch), k + 1), n)
        print(f"    IncrementalPCA: {combined.shape[1]} -> {k} dims (batch={batch})")
        ipca = IncrementalPCA(n_components=k, batch_size=batch)
        for start in range(0, n, batch):
            ipca.partial_fit(combined[start:start + batch])
        reduced = np.empty((n, k), dtype=np.float32)
        for start in range(0, n, batch):
            reduced[start:start + batch] = ipca.transform(
                combined[start:start + batch]
            ).astype(np.float32, copy=False)
        combined = reduced
        pca_dim_applied = int(k)
        pca_explained = float(np.sum(ipca.explained_variance_ratio_))
        print(f"    PCA explained variance ratio sum: {pca_explained:.4f}")

    coords = compute_umap(combined, args.n_neighbors, args.min_dist, args.seed)

    print("\n[4/4] Plotting + summary ...")
    panel_overlay(coords, labels, args.output_dir)

    np.savez_compressed(
        args.output_dir / "soap_umap_cache_generated_vs_qmof_splits.npz",
        coords=coords,
        labels=labels,
        cif_ids=all_ids,
    )

    summary = {
        "n_total_points": int(len(labels)),
        "n_generated": int(np.sum(labels == "generated")),
        "n_train": int(np.sum(labels == "train")),
        "n_val": int(np.sum(labels == "val")),
        "n_test": int(np.sum(labels == "test")),
        "split_labeled_ids_from_json": int(split_id_count),
        "qmof_rows_loaded": int(len(qmof_ids)),
        "qmof_rows_kept_in_splits": int(len(qmof_ids_split)),
        "qmof_rows_excluded_not_in_splits": int(n_unmatched_qmof),
        "generated_rows_loaded": int(len(generated_ids)),
        "soap_dim": int(feat_dim),
        "umap_input_dim": int(combined.shape[1]),
        "umap_params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
            "seed": args.seed,
        },
        "soap_params": {
            "r_cut": SOAP_RCUT,
            "n_max": SOAP_NMAX,
            "l_max": SOAP_LMAX,
            "sigma": SOAP_SIGMA,
            "periodic": SOAP_PERIODIC,
            "average": "inner",
        },
        "pca_before_umap": args.pca_dim > 0,
        "pca_dim_requested": int(args.pca_dim),
        "pca_dim_applied": pca_dim_applied,
        "pca_explained_variance_ratio_sum": pca_explained,
    }
    with (args.output_dir / "soap_generated_vs_qmof_splits_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"Done. Outputs in {args.output_dir}/")
    print("Panel: soap_umap_generated_vs_qmof_splits.(png|svg|pdf)")
    print("Cache: soap_umap_cache_generated_vs_qmof_splits.npz")
    print("Summary: soap_generated_vs_qmof_splits_summary.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
