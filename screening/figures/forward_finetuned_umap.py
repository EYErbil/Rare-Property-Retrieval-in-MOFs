#!/usr/bin/env python3
"""
Post-Training Embedding Extraction + UMAP — Fine-Tuned vs. Pretrained
======================================================================

Forward pass through the FINE-TUNED model on ALL MOFs (labeled + unlabeled),
then generate the same UMAP panels as Figure 2 for direct visual comparison.

This lets you see how the embedding space changed after training:
  - Did low-bandgap MOFs cluster more tightly?
  - Did the model pull similar structures closer together?
  - How do the unlabeled MOFs move?

Panels (saved individually):
  (a) Labeled vs. unlabeled
  (b) DFT bandgap — discrete bins (0–0.5, 0.5–1, 1–2, 2–3, 3–4, ≥4 eV)
  (c) Primary metal center (from qmof.csv)
  (d) Train / val / test split assignments

Two modes:
  1. --extract : Run forward pass with trained checkpoint → save NPZ
  2. --plot    : Load existing NPZ → generate UMAP panels
  (Can do both in one run: --extract --plot)

Usage:
  # Extract + plot in one go:
  python figures/forward_finetuned_umap.py \\
      --extract --plot \\
      --checkpoint /path/to/best_es-spearman=0.xxx.ckpt \\
      --data_dir data/raw/test \\
      --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --unlabeled_json data/unlabeled/test_bandgaps_regression.json \\
      --qmof_csv data/qmof.csv \\
      --output_dir figures_output/finetuned_umap_exp364_fulltune

  # Or from experiment name (auto-finds best checkpoint):
  python figures/forward_finetuned_umap.py \\
      --extract --plot \\
      --experiment exp364_fulltune \\
      --data_dir data/raw/test --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --unlabeled_json data/unlabeled/test_bandgaps_regression.json \\
      --qmof_csv data/qmof.csv --output_dir figures_output/finetuned_umap_exp364_fulltune

  # Plot-only from existing NPZ (fast re-runs to tweak visuals):
  python forward_finetuned_umap.py \\
      --plot \\
      --embeddings_npz figures_output/finetuned_umap_exp364_fulltune/posttrain_embeddings.npz \\
      --labeled_splits_dir data/splits/strategy_d_farthest_point --qmof_csv data/qmof.csv \\
      --output_dir figures_output/finetuned_umap_exp364_fulltune

Requirements: pip install moftransformer torch numpy matplotlib umap-learn
"""

import os
import sys
import json
import re
import csv as csvmod
import glob
import time
import shutil
import argparse
import numpy as np
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ══════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════
ORGANIC_ELEMENTS = frozenset({
    "C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I",
    "Si", "B", "Se", "Te", "As", "Ge", "Sb",
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
})

BANDGAP_BINS = [
    (0.0, 0.5,  "#d73027",  "0 – 0.5 eV"),
    (0.5, 1.0,  "#fc8d59",  "0.5 – 1 eV"),
    (1.0, 2.0,  "#fee08b",  "1 – 2 eV"),
    (2.0, 3.0,  "#d9ef8b",  "2 – 3 eV"),
    (3.0, 4.0,  "#91bfdb",  "3 – 4 eV"),
    (4.0, 99.0, "#4575b4",  "≥ 4 eV"),
]


# ══════════════════════════════════════════════════════════════════════
#  Publication style
# ══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
#  Part 1: Embedding extraction (from fine-tuned checkpoint)
# ══════════════════════════════════════════════════════════════════════

# --- data_dir resolution (MOFTransformer requires parent/split/ layout) ---
_VALID_SPLITS = {"train", "test", "val"}


def resolve_data_dir(data_dir):
    sample = [f for f in os.listdir(data_dir) if f.endswith(".graphdata")]
    if sample:
        parent = os.path.dirname(data_dir)
        basename = os.path.basename(data_dir)
        if basename in _VALID_SPLITS:
            return parent, basename, None
        link = os.path.join(parent, "test")
        if os.path.exists(link) or os.path.islink(link):
            real_link = os.path.realpath(link)
            real_data = os.path.realpath(data_dir)
            if real_link == real_data:
                return parent, "test", None
            else:
                raise RuntimeError(
                    f"{link} already exists but points to {real_link}, "
                    f"not {real_data}. Remove it and retry.")
        os.symlink(data_dir, link)
        return parent, "test", link
    for candidate in ("test", "train", "val"):
        if os.path.isdir(os.path.join(data_dir, candidate)):
            return data_dir, candidate, None
    return data_dir, "test", None


def discover_mof_ids(data_dir, split_subdir):
    search_dir = os.path.join(data_dir, split_subdir)
    ids = set()
    for fn in os.listdir(search_dir):
        if fn.endswith(".graphdata"):
            ids.add(fn[:-len(".graphdata")])
    return ids


def check_required_files(cid, data_dir, split_subdir):
    d = os.path.join(data_dir, split_subdir)
    for ext in (".graphdata", ".griddata16", ".grid"):
        if not os.path.exists(os.path.join(d, cid + ext)):
            return False
    return True


def build_unified_json(mof_ids, bg_map, data_dir, split_name,
                       downstream="bandgaps_regression"):
    labels = {cid: bg_map.get(cid, 0.0) for cid in sorted(mof_ids)}
    out_path = os.path.join(data_dir, f"{split_name}_{downstream}.json")
    backup = None
    if os.path.exists(out_path):
        backup = out_path + ".bak_posttrain"
        shutil.copy2(out_path, backup)
    with open(out_path, "w") as fh:
        json.dump(labels, fh)
    print(f"    Wrote unified JSON ({len(labels)} MOFs) -> {out_path}")
    return out_path, backup


def find_best_checkpoint(exp_dir):
    results_path = os.path.join(exp_dir, "final_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        best = results.get("checkpoints", {}).get("best", "")
        if best:
            if not os.path.isabs(best):
                best = os.path.join(exp_dir, best)
            if os.path.exists(best):
                return best
    for pattern in ["best_es-*.ckpt", "best_*.ckpt"]:
        matches = glob.glob(os.path.join(exp_dir, pattern))
        if matches:
            return sorted(matches)[-1]
    last = os.path.join(exp_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    return None


def extract_posttrain_embeddings(args):
    """Run forward pass with fine-tuned model on ALL MOFs → save NPZ."""
    import torch
    from torch.utils.data import DataLoader
    from moftransformer.datamodules.dataset import Dataset
    from moftransformer.config import config as default_config_fn
    from moftransformer.utils.validation import get_valid_config

    # Locate checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None and args.experiment:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        exp_dir = os.path.join(base_dir, "experiments", args.experiment)
        ckpt_path = find_best_checkpoint(exp_dir)
        if ckpt_path is None:
            sys.exit(f"ERROR: No checkpoint found for experiment {args.experiment}")
    if not ckpt_path or not os.path.exists(ckpt_path):
        sys.exit("ERROR: Provide --checkpoint or --experiment with a valid checkpoint")

    data_dir = os.path.abspath(args.data_dir)

    print(f"\n  Checkpoint : {ckpt_path}")
    print(f"  Data dir   : {data_dir}")

    # Resolve layout
    data_dir, split_name, _symlink = resolve_data_dir(data_dir)
    print(f"  Resolved   : {data_dir}/{split_name}/")

    # Load labels
    bg_map, split_map = {}, {}
    splits_dir = os.path.abspath(args.labeled_splits_dir)
    for sp in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{sp}_bandgaps_regression.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        for cid, bg in d.items():
            bg_map[cid] = float(bg)
            split_map[cid] = sp
        print(f"    {sp}: {len(d)} MOFs")
    labeled_cids = set(bg_map.keys())

    unlabeled_cids = set()
    if args.unlabeled_json and os.path.exists(args.unlabeled_json):
        with open(args.unlabeled_json) as fh:
            unlabeled_cids = set(json.load(fh).keys())
        print(f"    Unlabeled: {len(unlabeled_cids)}")

    # Discover MOFs on disk
    all_disk_ids = discover_mof_ids(data_dir, split_name)
    valid_ids = {cid for cid in all_disk_ids
                 if check_required_files(cid, data_dir, split_name)}
    n_lab_found = len(valid_ids & labeled_cids)
    n_ulab_found = len(valid_ids & unlabeled_cids)
    print(f"    On disk: {len(valid_ids)} valid "
          f"({n_lab_found} labeled, {n_ulab_found} unlabeled)")

    # Unified JSON
    json_path, backup_path = build_unified_json(
        valid_ids, bg_map, data_dir, split_name)

    # Build config
    config = default_config_fn()
    config = json.loads(json.dumps(config))
    config["data_dir"] = data_dir
    config["downstream"] = "bandgaps_regression"
    config["load_path"] = "pmtransformer"
    config["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "regression": 1, "classification": 0,
    }
    config["threshold"] = args.threshold
    config["pooling_type"] = "mean"
    config["dropout"] = 0.0
    config = get_valid_config(config)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # Load fine-tuned model
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(PROJECT_ROOT, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from train_regressor import MOFRegressor

    print("    Loading fine-tuned model from checkpoint ...")
    model = MOFRegressor.load_from_checkpoint(ckpt_path, config=config)
    model = model.to(device)
    model.eval()

    # Build dataset & loader for ALL MOFs
    ds = Dataset(
        data_dir, split=split_name,
        downstream=config["downstream"],
        nbr_fea_len=config.get("nbr_fea_len", 64),
        draw_false_grid=False,
    )
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda x: Dataset.collate(x, config.get("img_size", 30)),
        pin_memory=True,
    )

    total = len(ds)
    print(f"    Dataset: {total} MOFs")
    print("    Extracting embeddings ...")

    all_cid, all_emb = [], []
    n_done, n_skip = 0, 0
    t0 = time.time()
    last_report = t0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            try:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                # forward_features returns (pooled [B,768], output_dict)
                pooled, output = model.forward_features(batch)
                cids_batch = output.get("cif_id", output.get("name", None))
                cls_np = pooled.cpu().numpy()

                if cids_batch is not None:
                    if isinstance(cids_batch, (list, tuple)):
                        for i, cid in enumerate(cids_batch):
                            all_cid.append(str(cid))
                            all_emb.append(cls_np[i])
                    else:
                        all_cid.append(str(cids_batch))
                        all_emb.append(cls_np[0])
                else:
                    all_cid.append(f"batch_{batch_idx}")
                    all_emb.append(cls_np[0])
                n_done += cls_np.shape[0]
            except RuntimeError as e:
                err = str(e)
                if "shape" in err or "size" in err or "invalid" in err:
                    n_skip += 1
                    continue
                raise

            now = time.time()
            if now - last_report >= 30:
                last_report = now
                elapsed = now - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = (total - n_done) / rate if rate > 0 else 0
                eta_m, eta_s = divmod(int(remaining), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                print(f"      [{n_done:>6}/{total}]  "
                      f"{n_done/total*100:5.1f}%  "
                      f"ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s  "
                      f"({rate:.1f} MOF/s)  skipped {n_skip}",
                      flush=True)

    elapsed_total = time.time() - t0
    em, es = divmod(int(elapsed_total), 60)
    eh, em = divmod(em, 60)
    print(f"    Done: {n_done} embedded, {n_skip} skipped  "
          f"[{eh}h{em:02d}m{es:02d}s]")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Restore original JSON
    if backup_path and os.path.exists(backup_path):
        shutil.move(backup_path, json_path)
    elif os.path.exists(json_path):
        os.remove(json_path)

    if _symlink and os.path.islink(_symlink):
        os.unlink(_symlink)

    # Build metadata
    embeddings = np.stack(all_emb, axis=0) if all_emb else np.empty((0, 768))
    n = len(all_cid)
    bandgaps   = np.full(n, np.nan, dtype=np.float64)
    splits     = np.array(["unlabeled"] * n)
    is_labeled = np.zeros(n, dtype=bool)

    for i, cid in enumerate(all_cid):
        if cid in bg_map:
            bandgaps[i]   = bg_map[cid]
            splits[i]     = split_map.get(cid, "labeled")
            is_labeled[i] = True

    n_lab  = int(is_labeled.sum())
    n_ulab = n - n_lab
    print(f"    Final: {n} MOFs ({n_lab} labeled, {n_ulab} unlabeled), "
          f"dim={embeddings.shape[1]}")

    # Save NPZ
    npz_path = os.path.join(args.output_dir, "posttrain_embeddings.npz")
    np.savez_compressed(
        npz_path,
        cif_ids=np.array(all_cid),
        embeddings=embeddings,
        bandgaps=bandgaps,
        splits=splits,
        is_labeled=is_labeled,
    )
    print(f"    Saved → {npz_path}")
    return npz_path


# ══════════════════════════════════════════════════════════════════════
#  Part 2: UMAP Visualization (same panels as Figure 2)
# ══════════════════════════════════════════════════════════════════════

def load_npz(npz_path):
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


# --- Metal extraction (from qmof.csv) ---
def _metals_from_formula(formula):
    tokens = re.findall(r"([A-Z][a-z]?)\s*(\d*\.?\d*)", formula)
    metals = []
    for elem, cnt in tokens:
        if elem not in ORGANIC_ELEMENTS:
            metals.append((elem, float(cnt) if cnt else 1.0))
    return metals


def get_metals_from_qmof_csv(cif_ids, csv_path):
    name_to_formula, qmofid_to_formula = {}, {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csvmod.DictReader(fh)
        for row in reader:
            name = row.get("name", "").strip()
            formula = row.get("info.formula", "").strip()
            qmof_id = row.get("qmof_id", "").strip()
            if name and formula:
                name_to_formula[name] = formula
            if qmof_id and formula:
                qmofid_to_formula[qmof_id] = formula

    def _lookup(cid):
        if cid in name_to_formula:
            return name_to_formula[cid]
        bare = cid.replace(".cif", "")
        if bare in name_to_formula:
            return name_to_formula[bare]
        no_fsr = bare.replace("_FSR", "")
        if no_fsr in name_to_formula:
            return name_to_formula[no_fsr]
        with_fsr = bare + "_FSR"
        if with_fsr in name_to_formula:
            return name_to_formula[with_fsr]
        if bare in qmofid_to_formula:
            return qmofid_to_formula[bare]
        return None

    metals = {}
    for cid in cif_ids:
        formula = _lookup(cid)
        if not formula:
            metals[cid] = "Unknown"
            continue
        m = _metals_from_formula(formula)
        if m:
            m.sort(key=lambda x: -x[1])
            metals[cid] = m[0][0]
        else:
            metals[cid] = "Unknown"
    n_found = sum(1 for v in metals.values() if v != "Unknown")
    print(f"    Metal centers: {n_found}/{len(cif_ids)} identified")
    return metals


# --- Panels ---
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
    ax.set_title("(a)  Labeled vs. unlabeled  [fine-tuned]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "posttrain_a_labeled_unlabeled")
    plt.close()


def panel_b_bandgap(coords, is_labeled, bandgaps, output_dir):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    unlabeled_mask = ~is_labeled

    ax.scatter(coords[unlabeled_mask, 0], coords[unlabeled_mask, 1],
               c="#c0c0c0", s=1.5, alpha=0.30, rasterized=True, zorder=1)

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

    legend_handles.reverse()
    n_ulab = int(unlabeled_mask.sum())
    legend_handles.append(
        Line2D([], [], marker="o", color="w", markerfacecolor="#c0c0c0",
               markersize=5, label=f"Unlabeled ({n_ulab:,})"))

    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              fancybox=False, edgecolor="0.7", framealpha=0.95,
              borderpad=0.5, handletextpad=0.5, handlelength=1.4,
              fontsize=6, ncol=1)
    ax.set_title("(b)  DFT bandgap  [fine-tuned]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "posttrain_b_bandgap")
    plt.close()


def panel_c_metal(coords, metals, n_top_metals, output_dir):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    metal_counts = Counter(metals)
    top_metals = [m for m, _ in metal_counts.most_common()
                  if m != "Unknown"][:n_top_metals]

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
    ax.set_title("(c)  Primary metal center  [fine-tuned]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "posttrain_c_metal_center")
    plt.close()


def panel_d_splits(coords, is_labeled, bandgaps, split_labels,
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
    ax.set_title("(d)  Train / val / test  [fine-tuned]",
                 fontweight="bold", pad=6, fontsize=9)
    _style_ax(ax)
    plt.tight_layout()
    _save_panel(fig, output_dir, "posttrain_d_splits")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(
        description="Post-Training Embedding Extraction + UMAP",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Mode
    pa.add_argument("--extract", action="store_true",
                    help="Run forward pass to extract embeddings")
    pa.add_argument("--plot", action="store_true",
                    help="Generate UMAP panels")

    # Extraction args
    g1 = pa.add_argument_group("Extraction (--extract)")
    g1.add_argument("--checkpoint", default=None,
                    help="Path to fine-tuned .ckpt file")
    g1.add_argument("--experiment", default=None,
                    help="Experiment name (auto-finds best checkpoint)")
    g1.add_argument("--data_dir", default=None,
                    help="Dir with .graphdata/.griddata16/.grid for ALL MOFs")
    g1.add_argument("--unlabeled_json", default=None,
                    help="Unlabeled test_bandgaps_regression.json")
    g1.add_argument("--device", default=None)

    # Plot args
    g2 = pa.add_argument_group("Plotting (--plot)")
    g2.add_argument("--embeddings_npz", default=None,
                    help="Pre-extracted NPZ (skip extraction, plot only)")
    g2.add_argument("--qmof_csv", default=None,
                    help="qmof.csv for metal detection")
    g2.add_argument("--n_top_metals", type=int, default=8)
    g2.add_argument("--load_umap_cache", default=None)
    g2.add_argument("--save_umap_cache", action="store_true")

    # Shared
    pa.add_argument("--labeled_splits_dir", required=True,
                    help="Dir with {train,val,test}_bandgaps_regression.json")
    pa.add_argument("--output_dir", default="./figures_output/finetuned_umap")
    pa.add_argument("--threshold", type=float, default=1.0)
    pa.add_argument("--n_neighbors", type=int, default=30)
    pa.add_argument("--min_dist", type=float, default=0.3)
    pa.add_argument("--seed", type=int, default=42)

    args = pa.parse_args()

    if not args.extract and not args.plot:
        sys.exit("ERROR: specify --extract and/or --plot")

    os.makedirs(args.output_dir, exist_ok=True)
    set_publication_style()

    print("=" * 70)
    print("  POST-TRAINING UMAP — Fine-Tuned Model Embedding Space")
    print("=" * 70)

    npz_path = None

    # ── EXTRACTION ────────────────────────────────────────────────────
    if args.extract:
        if not args.data_dir:
            sys.exit("ERROR: --extract requires --data_dir")
        print("\n[EXTRACT] Running forward pass with fine-tuned model ...")
        npz_path = extract_posttrain_embeddings(args)

    # ── PLOTTING ──────────────────────────────────────────────────────
    if args.plot:
        if args.embeddings_npz:
            npz_path = args.embeddings_npz
        elif npz_path is None:
            npz_path = os.path.join(args.output_dir, "posttrain_embeddings.npz")

        if not os.path.exists(npz_path):
            sys.exit(f"ERROR: NPZ not found: {npz_path}\n"
                     f"  Run with --extract first, or provide --embeddings_npz")

        print(f"\n[PLOT] Loading embeddings from {npz_path} ...")
        cids, embs, bgs_raw, splits_raw, is_lab_raw = load_npz(npz_path)

        # Load authoritative labels
        labeled_bg, labeled_splits_dict = load_split_labels(
            os.path.abspath(args.labeled_splits_dir))
        labeled_cid_set = set(labeled_bg.keys())

        # Build arrays
        if is_lab_raw is not None:
            is_labeled = is_lab_raw
        else:
            is_labeled = np.array([c in labeled_cid_set for c in cids])

        bgs = np.array([labeled_bg.get(c, float(bgs_raw[i]))
                        for i, c in enumerate(cids)], dtype=float)
        split_labels = np.array([labeled_splits_dict.get(c, "unlabeled")
                                 for c in cids])

        # UMAP
        if args.load_umap_cache and os.path.exists(args.load_umap_cache):
            print(f"\n  Loading cached UMAP coords: {args.load_umap_cache}")
            cache = np.load(args.load_umap_cache, allow_pickle=True)
            coords = cache["coords"]
            cached_ids = [str(x) for x in cache["cif_ids"]]
            if cached_ids != cids:
                print("  WARNING: cached IDs differ — recomputing")
                coords = compute_umap(embs, args.n_neighbors, args.min_dist,
                                      seed=args.seed)
            else:
                print(f"  {coords.shape[0]} cached points loaded")
        else:
            print(f"\n  Computing UMAP ...")
            coords = compute_umap(embs, args.n_neighbors, args.min_dist,
                                  seed=args.seed)

        if args.save_umap_cache:
            cp = os.path.join(args.output_dir, "posttrain_umap_cache.npz")
            np.savez_compressed(cp, coords=coords, cif_ids=np.array(cids))
            print(f"  Cache saved: {cp}")

        # Metal centers
        if args.qmof_csv and os.path.exists(args.qmof_csv):
            metals_dict = get_metals_from_qmof_csv(cids, args.qmof_csv)
        else:
            metals_dict = {c: "Unknown" for c in cids}
        metals = np.array([metals_dict.get(c, "Unknown") for c in cids])

        # Filter invalid
        valid = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
        norms = np.linalg.norm(embs, axis=1)
        valid &= norms > 1e-6
        n_skip = int((~valid).sum())
        if n_skip > 0:
            print(f"  Skipping {n_skip} MOFs with invalid embeddings/coords")
            coords       = coords[valid]
            is_labeled   = is_labeled[valid]
            bgs          = bgs[valid]
            metals       = metals[valid]
            split_labels = split_labels[valid]
            cids         = [c for c, v in zip(cids, valid) if v]

        n_lab = int(is_labeled.sum())
        n_ulab = len(cids) - n_lab
        print(f"\n  Final: {len(cids)} MOFs ({n_lab} labeled, {n_ulab} unlabeled)")

        # Generate panels
        print(f"\n  Generating panels ...")
        panel_a_labeled_unlabeled(coords, is_labeled, args.output_dir)
        panel_b_bandgap(coords, is_labeled, bgs, args.output_dir)
        panel_c_metal(coords, metals, args.n_top_metals, args.output_dir)
        panel_d_splits(coords, is_labeled, bgs, split_labels,
                       args.threshold, args.output_dir)

        # Summary
        summary = {
            "total_mofs": len(cids),
            "labeled": n_lab,
            "unlabeled": n_ulab,
            "embedding_dim": int(embs.shape[1]),
            "checkpoint": args.checkpoint or args.experiment or "unknown",
            "umap_params": {
                "n_neighbors": args.n_neighbors,
                "min_dist": args.min_dist,
                "metric": "cosine",
                "seed": args.seed,
            },
        }
        sp = os.path.join(args.output_dir, "posttrain_umap_summary.json")
        with open(sp, "w") as fh:
            json.dump(summary, fh, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Done.  All outputs in {args.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
