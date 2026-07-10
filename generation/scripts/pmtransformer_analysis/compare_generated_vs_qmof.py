#!/usr/bin/env python3
"""
PMTransformer Comparison - generated-dataset vs QMOF
===============================================

Extracts 768-dim CLS embeddings from pretrained PMTransformer for both the
newly generated MOFs (label = "generated-dataset") and QMOF (label = "QMOF"),
then projects them into a single 2D UMAP so we can see how much the generated
set covers the QMOF manifold.

The extraction pipeline, config, and .npz layout are kept identical to
``extract_all_embeddings_unified.py``:
    - Module(config) with config["load_path"] = "pmtransformer"
    - Dataset(data_dir, split=..., downstream=..., draw_false_grid=False)
    - output["cls_feats"]  (768-dim CLS token, one vector per MOF)
    - Saves {cif_ids, embeddings} as .npz

Two input modes per dataset
---------------------------
  --qmof-embeddings / --generated-embeddings
      Reuse an existing .npz (keys cif_ids + embeddings). Typical source:
      all_embeddings.npz from extract_all_embeddings_unified.py.

  --qmof-data-dir / --generated-data-dir
      Run pretrained PMTransformer forward (no fine-tuning) from a directory
      that contains the preprocessed .graphdata / .griddata16 / .grid files.
      Produce those files first with ``moftransformer.utils.prepare_data`` (see
      ``scripts/prepare_moftransformer_test_only.py`` in this repo).
      or point at an existing Train_ready-style layout.

  --labeled-splits-dir (optional)
      Directory with train/val/test_bandgaps_regression.json. When set, the
      script writes **additional** split-colored panels that reuse the **same**
      2D UMAP coordinates as the main plot: only split-labeled QMOF points +
      all generated are **drawn** (unlabeled / out-of-split QMOF are omitted
      from that figure only). One UMAP fit total.

Outputs (in --output_dir):
    qmof_pmt_embeddings.npz
    generated_pmt_embeddings.npz
    pmt_umap_generated_vs_qmof.(png|svg|pdf)
    pmt_umap_density_generated_vs_qmof.(png|svg|pdf)
    pmt_comparison_summary.json
    (if --labeled-splits-dir) pmt_umap_generated_vs_qmof_splits.(png|svg|pdf)
    (if --labeled-splits-dir) pmt_umap_density_generated_vs_qmof_splits.(png|svg|pdf)

Usage (all cached; fastest path)
--------------------------------
    python scripts/pmtransformer_analysis/compare_generated_vs_qmof.py \\
        --qmof-embeddings embeddings/pmt_embeddings_qmof_all.npz \\
        --generated-embeddings  generated_embeddings/all_embeddings.npz

Usage (compute generated embeddings from scratch)
--------------------------------
    python scripts/pmtransformer_analysis/compare_generated_vs_qmof.py \\
        --qmof-embeddings embeddings/pmt_embeddings_qmof_all.npz \\
        --generated-data-dir    generated_preprocessed/ \\
        --generated-split       test

Requirements: moftransformer, torch, numpy, matplotlib, umap-learn, scipy.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------- #
# Publication style - same helpers as figure_soap_umap.py so the figures
# produced here look visually identical to the SOAP ones.
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
# PMTransformer extraction - mirrors extract_all_embeddings_unified.py:
#   * config["load_path"] = "pmtransformer"
#   * loss_names has regression=1 (we don't use the head, but config validation
#     expects at least one loss)
#   * output["cls_feats"] -> 768-dim CLS vector per MOF
#   * a dummy <split>_<downstream>.json is written so Dataset can load all
#     preprocessed MOFs in a single pass
# --------------------------------------------------------------------------- #
_VALID_SPLITS = {"train", "test", "val"}


def resolve_data_dir(data_dir: Path) -> tuple[Path, str, Path | None]:
    """Accept either a parent folder containing a train/val/test subdir, or a
    leaf folder with .graphdata files inside. Returns (data_dir, split,
    symlink_to_clean). Copied (slightly simplified) from
    extract_all_embeddings_unified.py."""
    data_dir = data_dir.resolve()
    sample = [p for p in data_dir.iterdir() if p.suffix == ".graphdata"]
    if sample:
        parent = data_dir.parent
        if data_dir.name in _VALID_SPLITS:
            return parent, data_dir.name, None
        link = parent / "test"
        if link.exists() or link.is_symlink():
            real_link = link.resolve()
            if real_link == data_dir:
                return parent, "test", None
            raise RuntimeError(f"{link} exists but points to {real_link}; resolve manually.")
        os.symlink(str(data_dir), str(link))
        print(f"    Created symlink {link} -> {data_dir}")
        return parent, "test", link
    for candidate in ("test", "train", "val"):
        if (data_dir / candidate).is_dir():
            return data_dir, candidate, None
    return data_dir, "test", None


def discover_mof_ids(data_dir: Path, split_subdir: str) -> set[str]:
    search = data_dir / split_subdir
    return {p.stem for p in search.iterdir() if p.suffix == ".graphdata"}


def check_required_files(cid: str, data_dir: Path, split_subdir: str) -> bool:
    d = data_dir / split_subdir
    return all((d / f"{cid}{ext}").exists()
               for ext in (".graphdata", ".griddata16", ".grid"))


def write_unified_label_json(
    mof_ids: set[str], data_dir: Path, split_name: str, downstream: str,
) -> tuple[Path, Path | None]:
    """Write a dummy labels JSON listing every MOF so Dataset can iterate
    over all of them. Returns (json_path, backup_path_or_None)."""
    out = data_dir / f"{split_name}_{downstream}.json"
    backup = None
    if out.exists():
        backup = out.with_name(out.name + ".bak_generated_vs_qmof")
        shutil.copy2(out, backup)
    with out.open("w") as fh:
        json.dump({cid: 0.0 for cid in sorted(mof_ids)}, fh)
    print(f"    Wrote unified label JSON ({len(mof_ids)} MOFs) -> {out}")
    return out, backup


def _extract_embeddings(
    data_dir: Path,
    split_name: str,
    downstream: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    """Single forward pass over every preprocessed MOF under
    data_dir/<split>/.  Returns (cif_ids, [N, 768] embeddings)."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from moftransformer.modules.module import Module
        from moftransformer.datamodules.dataset import Dataset
        from moftransformer.config import config as default_config_fn
        from moftransformer.utils.validation import get_valid_config
    except ImportError as exc:
        sys.exit(
            "Could not import moftransformer/torch. Install with\n"
            "    pip install moftransformer\n"
            "and ensure your CIFs were preprocessed into "
            ".graphdata / .griddata16 / .grid via\n"
            "    moftransformer.utils.prepare_data.prepare_data(...)\n"
            f"Original error: {exc}"
        )

    config = default_config_fn()
    config = json.loads(json.dumps(config))  # deep-copy via JSON
    config["data_dir"]   = str(data_dir)
    config["downstream"] = downstream
    config["load_path"]  = "pmtransformer"
    config["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "regression": 1, "classification": 0,
    }
    config = get_valid_config(config)

    device_obj = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"    Device: {device_obj}")

    model = Module(config)
    model.eval().to(device_obj)

    ds = Dataset(
        str(data_dir),
        split=split_name,
        downstream=downstream,
        nbr_fea_len=config.get("nbr_fea_len", 64),
        draw_false_grid=False,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: Dataset.collate(x, config.get("img_size", 30)),
        pin_memory=True,
    )
    total = len(ds)
    print(f"    Dataset size: {total}")

    all_cid: list[str] = []
    all_emb: list[np.ndarray] = []
    n_done = n_skip = 0
    t0 = time.time()
    last_report = t0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            try:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device_obj)
                out = model.infer(batch)
                cls = out["cls_feats"].cpu().numpy()
                for i, cid in enumerate(out["cif_id"]):
                    all_cid.append(cid)
                    all_emb.append(cls[i])
                n_done += len(out["cif_id"])
            except RuntimeError as e:
                msg = str(e)
                if "shape" in msg or "invalid" in msg or "size" in msg:
                    n_skip += 1
                    continue
                raise

            now = time.time()
            if now - last_report >= 30:
                last_report = now
                rate = n_done / (now - t0) if (now - t0) > 0 else 0
                remaining = (total - n_done) / rate if rate > 0 else 0
                em, es = divmod(int(remaining), 60)
                eh, em = divmod(em, 60)
                print(f"      [{n_done:>6}/{total}]  "
                      f"{n_done/total*100:5.1f}%  "
                      f"ETA {eh}h{em:02d}m{es:02d}s  "
                      f"({rate:.1f} MOF/s)  skipped {n_skip}",
                      flush=True)

    del model
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    em, es = divmod(int(elapsed), 60)
    eh, em = divmod(em, 60)
    print(f"    Extraction done: {n_done} embedded, {n_skip} skipped  "
          f"[{eh}h{em:02d}m{es:02d}s]")

    emb = np.stack(all_emb, axis=0) if all_emb else np.empty((0, 768))
    return all_cid, emb


def compute_pmtransformer_embeddings(
    data_dir: Path,
    split: str | None,
    downstream: str,
    batch_size: int,
    num_workers: int,
    device: str,
    output_cache: Path,
) -> tuple[list[str], np.ndarray]:
    """Full pipeline: resolve layout -> write unified JSON -> extract ->
    restore JSON -> save .npz."""
    resolved_dir, split_name, symlink = resolve_data_dir(data_dir)
    if split is not None:
        split_name = split
    print(f"    Resolved data_dir: {resolved_dir}")
    print(f"    Split sub-folder : {split_name}/")

    mof_ids = {cid for cid in discover_mof_ids(resolved_dir, split_name)
               if check_required_files(cid, resolved_dir, split_name)}
    print(f"    Valid MOFs (all three preprocessed files present): {len(mof_ids)}")
    if not mof_ids:
        sys.exit(f"No usable MOFs found in {resolved_dir / split_name}")

    json_path, backup = write_unified_label_json(
        mof_ids, resolved_dir, split_name, downstream)

    try:
        cif_ids, emb = _extract_embeddings(
            resolved_dir, split_name, downstream,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
    finally:
        # Always restore the data dir to its original state
        if backup and backup.exists():
            shutil.move(str(backup), str(json_path))
        elif json_path.exists():
            json_path.unlink()
        if symlink and symlink.is_symlink():
            symlink.unlink()

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_cache,
        cif_ids=np.array(cif_ids),
        embeddings=emb,
    )
    print(f"    Saved {len(cif_ids)} embeddings (dim={emb.shape[1]}) -> {output_cache}")
    return cif_ids, emb


# --------------------------------------------------------------------------- #
# Load existing .npz (supports the extract_all_embeddings_unified.py format)
# --------------------------------------------------------------------------- #
def load_embeddings_cache(cache_path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(cache_path, allow_pickle=True)
    cif_ids = [str(c) for c in data["cif_ids"]]
    key = None
    for k in ("embeddings", "descriptors", "emb", "X"):
        if k in data.files:
            key = k
            break
    if key is None:
        raise KeyError(f"{cache_path} has no embedding array; keys={data.files}")
    emb = np.asarray(data[key], dtype=np.float32)
    print(f"    Loaded cache {cache_path}: {len(cif_ids)} MOFs, dim={emb.shape[1]} "
          f"(key='{key}')")
    return cif_ids, emb


# --------------------------------------------------------------------------- #
# QMOF train/val/test splits (same JSON convention as SOAP split script)
# --------------------------------------------------------------------------- #
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


def split_panel_row_indices_and_labels(
    qmof_ids_all: list[str],
    generated_ids: list[str],
    split_assignments: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices into stacked [qmof; generated] UMAP rows for split-colored panels.
    Includes every generated row and every QMOF row whose id maps to train/val/test.
    """
    keep: list[int] = []
    labels: list[str] = []
    n_q = len(qmof_ids_all)
    for i, cid in enumerate(qmof_ids_all):
        sp = _flex_lookup(cid, split_assignments)
        if sp in ("train", "val", "test"):
            keep.append(i)
            labels.append(sp)
    if not keep:
        sys.exit("No QMOF rows matched train/val/test split assignments.")
    for j in range(len(generated_ids)):
        keep.append(n_q + j)
        labels.append("generated")
    return (
        np.asarray(keep, dtype=np.int64),
        np.asarray(labels, dtype=object),
    )


# --------------------------------------------------------------------------- #
# UMAP + plotting (mirrors SOAP script)
# --------------------------------------------------------------------------- #
def compute_umap(matrix: np.ndarray, n_neighbors: int, min_dist: float,
                 seed: int) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError:
        sys.exit("ERROR: umap-learn not installed. pip install umap-learn")
    n, d = matrix.shape[0], matrix.shape[1]
    print(f"    {n} points, dim={d} ...")
    n_jobs = 1 if (d > 2000 or n > 20000) else -1
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        n_jobs=n_jobs,
        low_memory=True,
    )
    coords = reducer.fit_transform(matrix)
    print(f"    UMAP done -> {coords.shape}")
    return coords


def panel_dataset_overlay(coords: np.ndarray, labels: np.ndarray,
                          output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    qmask = labels == "QMOF"
    emask = labels == "generated-dataset"
    n_qmof = int(qmask.sum())
    n_generated = int(emask.sum())

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
    ax.set_title("PMTransformer UMAP: generated-dataset vs. QMOF",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "pmt_umap_generated_vs_qmof")
    plt.close()


def panel_density_contour(coords: np.ndarray, labels: np.ndarray,
                          output_dir: Path) -> None:
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

    fig.suptitle("PMTransformer UMAP density - where each dataset concentrates",
                 fontweight="bold", fontsize=9, y=1.01)
    plt.tight_layout()
    _save_panel(fig, output_dir, "pmt_umap_density_generated_vs_qmof")
    plt.close()


def panel_splits_overlay(
    coords: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    base_name: str = "pmt_umap_generated_vs_qmof_splits",
) -> None:
    """Train / val / test (QMOF) + generated — same color scheme as SOAP split panel."""
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
    ax.set_title(
        "PMTransformer UMAP: Generated vs QMOF train/val/test",
        fontweight="bold",
        pad=6,
        fontsize=9,
    )
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, base_name)
    plt.close()


def panel_splits_density(
    coords: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    base_name: str = "pmt_umap_density_generated_vs_qmof_splits",
) -> None:
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("    scipy not available - skipping split density panel.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.0), sharex=True, sharey=True)
    panels = [
        ("train", "#4292c6"),
        ("val", "#fd8d3c"),
        ("test", "#969696"),
        ("generated", "#e41a1c"),
    ]
    for ax, (name, color) in zip(axes.flat, panels):
        mask = labels == name
        n_pts = int(mask.sum())
        ax.scatter(
            coords[:, 0], coords[:, 1], c="#d9d9d9", s=1.0,
            alpha=0.30, rasterized=True, zorder=1,
        )
        sub = coords[mask]
        if sub.shape[0] >= 20:
            kde = gaussian_kde(sub.T)
            xg = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
            yg = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
            X, Y = np.meshgrid(xg, yg)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=6, colors=color, linewidths=0.55, zorder=3)
        if n_pts:
            ax.scatter(sub[:, 0], sub[:, 1], c=color, s=1.5, alpha=0.55,
                       rasterized=True, zorder=2)
        ax.set_title(f"{name.capitalize()} ({n_pts:,})", fontweight="bold", fontsize=9)
        _style_ax(ax)

    fig.suptitle(
        "PMTransformer UMAP density by split / generated",
        fontweight="bold",
        fontsize=9,
        y=1.01,
    )
    plt.tight_layout()
    _save_panel(fig, output_dir, base_name)
    plt.close()


# --------------------------------------------------------------------------- #
# Per-dataset prep (either load cache or run extraction)
# --------------------------------------------------------------------------- #
def prepare_dataset(
    label: str,
    data_dir: Path | None,
    cache_path: Path | None,
    split: str | None,
    downstream: str,
    batch_size: int,
    num_workers: int,
    device: str,
    output_cache: Path,
    limit: int,
) -> tuple[list[str], np.ndarray]:
    if cache_path is not None and cache_path.exists():
        cif_ids, emb = load_embeddings_cache(cache_path)
        if limit > 0 and len(cif_ids) > limit:
            cif_ids = cif_ids[:limit]
            emb = emb[:limit]
            print(f"    [{label}] capped to {limit} cached rows")
        return cif_ids, emb

    assert data_dir is not None, f"[{label}] need --{label}-data-dir or --{label}-embeddings"
    cif_ids, emb = compute_pmtransformer_embeddings(
        data_dir=data_dir,
        split=split,
        downstream=downstream,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        output_cache=output_cache,
    )
    if limit > 0 and len(cif_ids) > limit:
        cif_ids = cif_ids[:limit]
        emb = emb[:limit]
    return cif_ids, emb


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    pa = argparse.ArgumentParser(
        description="PMTransformer comparison of generated generated-dataset against QMOF",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    pa.add_argument("--qmof-data-dir", type=Path, default=None,
                    help="Preprocessed QMOF dir (parent or leaf with .graphdata/.griddata16/.grid).")
    pa.add_argument("--generated-data-dir", type=Path, default=None,
                    help="Preprocessed generated-dataset dir with moftransformer inputs.")
    pa.add_argument(
        "--qmof-embeddings", "--qmof-cache",
        type=Path,
        default=None,
        dest="qmof_embeddings",
        help="Cached .npz with cif_ids + embeddings (e.g. all_embeddings.npz). Alias: --qmof-cache.",
    )
    pa.add_argument(
        "--generated-embeddings", "--generated-cache",
        type=Path,
        default=None,
        dest="generated_embeddings",
        help="Cached .npz with cif_ids + embeddings for generated MOFs. Alias: --generated-cache.",
    )
    pa.add_argument(
        "--labeled-splits-dir",
        type=Path,
        default=None,
        help=(
            "Optional. Dir with {train,val,test}_bandgaps_regression.json. "
            "Adds split-colored panels using the **same** UMAP coordinates as the "
            "main plot; only split-labeled QMOF + all generated are drawn."
        ),
    )
    pa.add_argument("--qmof-split", default=None,
                    help="Optional override for the split sub-folder name under qmof-data-dir.")
    pa.add_argument("--generated-split", default=None,
                    help="Optional override for the split sub-folder name under generated-data-dir.")
    pa.add_argument("--max-qmof", type=int, default=0)
    pa.add_argument("--max-generated", type=int, default=0)
    pa.add_argument("--downstream", default="bandgaps_regression",
                    help="moftransformer downstream tag; only affects the dummy JSON name.")
    pa.add_argument("--batch_size", type=int, default=1)
    pa.add_argument("--num_workers", type=int, default=0)
    pa.add_argument("--device", default=None,
                    help="Torch device; default auto (cuda if available).")
    pa.add_argument("--output_dir", type=Path,
                    default=Path("pmtransformer_analysis/generated_vs_qmof"))
    pa.add_argument("--n_neighbors", type=int, default=30)
    pa.add_argument("--min_dist", type=float, default=0.3)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--save_umap_cache", action="store_true")
    args = pa.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    device = args.device or ("cuda" if _cuda_available() else "cpu")

    print("=" * 70)
    print("  PMTRANSFORMER COMPARISON - generated-dataset vs QMOF")
    print("=" * 70)

    if args.qmof_embeddings is None and args.qmof_data_dir is None:
        pa.error("Need --qmof-embeddings or --qmof-data-dir")
    if args.generated_embeddings is None and args.generated_data_dir is None:
        pa.error("Need --generated-embeddings or --generated-data-dir")

    print("\n[1/3] Gathering PMTransformer embeddings ...")
    qmof_ids_all, qmof_emb_all = prepare_dataset(
        "qmof",
        args.qmof_data_dir, args.qmof_embeddings,
        args.qmof_split, args.downstream,
        args.batch_size, args.num_workers, device,
        args.output_dir / "qmof_pmt_embeddings.npz",
        args.max_qmof,
    )

    generated_ids, generated_emb = prepare_dataset(
        "generated",
        args.generated_data_dir, args.generated_embeddings,
        args.generated_split, args.downstream,
        args.batch_size, args.num_workers, device,
        args.output_dir / "generated_pmt_embeddings.npz",
        args.max_generated,
    )

    if qmof_emb_all.size == 0 or generated_emb.size == 0:
        sys.exit("One of the embedding sets is empty - aborting.")
    if qmof_emb_all.shape[1] != generated_emb.shape[1]:
        sys.exit(
            f"PMTransformer dim mismatch: QMOF={qmof_emb_all.shape[1]} vs "
            f"generated={generated_emb.shape[1]}. Both must come from the same checkpoint."
        )

    emb_dim = int(qmof_emb_all.shape[1])
    n_q_all = len(qmof_ids_all)
    n_e = len(generated_ids)
    print(f"\n    Full QMOF cache: {n_q_all} rows; generated: {n_e} rows "
          f"(embedding dim = {emb_dim})")

    # ----- Single UMAP: ALL QMOF (labeled + unlabeled) + generated -----
    print("\n[2/3] UMAP: all QMOF + generated ...")
    combined_all = np.empty((n_q_all + n_e, emb_dim), dtype=np.float32)
    combined_all[:n_q_all] = np.asarray(qmof_emb_all, dtype=np.float32)
    combined_all[n_q_all:] = np.asarray(generated_emb, dtype=np.float32)

    labels_all = np.array(["QMOF"] * n_q_all + ["generated-dataset"] * n_e)
    cif_ids_all = np.array(qmof_ids_all + generated_ids)

    coords_all = compute_umap(
        combined_all, args.n_neighbors, args.min_dist, args.seed,
    )
    del combined_all
    gc.collect()

    panel_dataset_overlay(coords_all, labels_all, args.output_dir)
    panel_density_contour(coords_all, labels_all, args.output_dir)

    if args.save_umap_cache:
        cp = args.output_dir / "pmt_umap_cache.npz"
        np.savez_compressed(
            cp, coords=coords_all, labels=labels_all, cif_ids=cif_ids_all,
        )
        print(f"    UMAP cache (all QMOF) saved -> {cp}")

    # ----- Split-colored panels: same UMAP coords, subset of rows -----
    labels_split: np.ndarray | None = None
    coords_split: np.ndarray | None = None
    cif_ids_split_arr: np.ndarray | None = None
    n_q_split = 0
    n_excluded_from_splits: int | None = None

    if args.labeled_splits_dir is not None:
        print("\n[3/3] Split-colored panels (same UMAP; labeled QMOF + generated only) ...")
        split_assignments = load_split_assignments(args.labeled_splits_dir.resolve())
        row_idx, labels_split = split_panel_row_indices_and_labels(
            qmof_ids_all, generated_ids, split_assignments,
        )
        n_q_split = int(len(row_idx) - n_e)
        n_excluded_from_splits = n_q_all - n_q_split
        print(
            f"    QMOF rows drawn in split panels: {n_q_split} "
            f"(omitted from split figure only: {n_excluded_from_splits})"
        )

        coords_split = coords_all[row_idx]
        cif_ids_split_arr = cif_ids_all[row_idx]

        panel_splits_overlay(coords_split, labels_split, args.output_dir)
        panel_splits_density(coords_split, labels_split, args.output_dir)

        if args.save_umap_cache:
            cps = args.output_dir / "pmt_umap_cache_splits.npz"
            np.savez_compressed(
                cps,
                coords=coords_split,
                labels=labels_split,
                cif_ids=cif_ids_split_arr,
                umap_row_index=row_idx,
            )
            print(f"    Split-panel point cache saved -> {cps}")
    else:
        print("\n[3/3] Skipping split panels (--labeled-splits-dir not set).")

    del qmof_emb_all, generated_emb
    gc.collect()

    summary = {
        "umap_all_qmof_plus_generated": {
            "n_qmof": int(n_q_all),
            "n_generated": int(n_e),
            "n_total": int(n_q_all + n_e),
        },
        "umap_split_labeled_qmof_plus_generated": None,
        "split_panels_use_same_umap_coordinates": bool(args.labeled_splits_dir),
        "labeled_splits_dir": str(args.labeled_splits_dir.resolve())
        if args.labeled_splits_dir else None,
        "embedding_dim": emb_dim,
        "pmt_config": {
            "load_path": "pmtransformer",
            "downstream": args.downstream,
        },
        "umap_params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
            "seed": args.seed,
        },
    }
    if labels_split is not None and coords_split is not None:
        summary["umap_split_labeled_qmof_plus_generated"] = {
            "n_qmof_labeled_drawn": int(n_q_split),
            "n_generated": int(n_e),
            "n_total_drawn": int(n_q_split + n_e),
            "n_qmof_omitted_from_split_figure_only": int(n_excluded_from_splits or 0),
            "n_train": int(np.sum(labels_split == "train")),
            "n_val": int(np.sum(labels_split == "val")),
            "n_test": int(np.sum(labels_split == "test")),
            "n_generated": int(np.sum(labels_split == "generated")),
        }
    with (args.output_dir / "pmt_comparison_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Done. All outputs in {args.output_dir}/")
    print("  (1) All QMOF + generated: pmt_umap_generated_vs_qmof.(png|svg|pdf), "
          "pmt_umap_density_generated_vs_qmof.(png|svg|pdf)")
    if labels_split is not None:
        print("  (2) Same UMAP, labeled QMOF + generated only: "
              "pmt_umap_generated_vs_qmof_splits.(png|svg|pdf), "
              "pmt_umap_density_generated_vs_qmof_splits.(png|svg|pdf)")
    print(f"{'=' * 70}")
    return 0


def _cuda_available() -> bool:
    """Check CUDA without hard-importing torch at module top so cached-only
    runs can work even without torch installed."""
    try:
        import torch  # noqa: WPS433
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
