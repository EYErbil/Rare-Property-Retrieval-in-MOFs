#!/usr/bin/env python3
"""
Unified Embedding Extraction — Single Forward Pass for ALL MOFs
================================================================

Runs pretrained PMTransformer (MOFTransformer) once on every MOF (labeled +
unlabeled) to guarantee all embeddings live in exactly the same space.

This addresses the alignment risk that comes from extracting labeled and
unlabeled embeddings in two separate runs with potentially different code.

Output
------
  figures_output/pretrained_embeddings/all_embeddings.npz
    Keys:
      cif_ids     – np.array of str (with _FSR suffix)
      embeddings  – np.ndarray [N, 768]
      bandgaps    – np.ndarray [N]  (DFT value for labeled, NaN for unlabeled)
      splits      – np.array of str ('train','val','test' for labeled; 'unlabeled')
      is_labeled  – np.array of bool

Usage
-----
  python figures/forward_pretrained_embeddings.py \\
      --data_dir data/raw/test \\
      --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --unlabeled_json data/unlabeled/test_bandgaps_regression.json \\
      --output_dir figures_output/pretrained_embeddings

Requirements
------------
  pip install moftransformer torch numpy
"""

import os
import sys
import json
import time
import argparse
import shutil
import tempfile
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from moftransformer.modules.module import Module
from moftransformer.datamodules.dataset import Dataset
from moftransformer.config import config as default_config_fn
from moftransformer.utils.validation import get_valid_config


# ══════════════════════════════════════════════════════════════════════
#  Label loading
# ══════════════════════════════════════════════════════════════════════
def load_labeled_splits(splits_dir, downstream="bandgaps_regression"):
    """Return (cif_id → bandgap, cif_id → split_name) from split JSONs."""
    bg_map, split_map = {}, {}
    for split in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{split}_{downstream}.json")
        if not os.path.exists(p):
            print(f"  WARNING: {p} not found, skipping {split}")
            continue
        with open(p) as fh:
            d = json.load(fh)
        for cid, val in d.items():
            bg_map[cid] = float(val)
            split_map[cid] = split
        print(f"  {split}: {len(d)} MOFs")
    return bg_map, split_map


def load_unlabeled_ids(json_path):
    """Load set of unlabeled CIF IDs from unlabeled JSON."""
    with open(json_path) as fh:
        return set(json.load(fh).keys())


# ══════════════════════════════════════════════════════════════════════
#  Discover ALL MOFs in data_dir
# ══════════════════════════════════════════════════════════════════════
# MOFTransformer hard-codes: assert split in {"train", "test", "val"}
_VALID_SPLITS = {"train", "test", "val"}


def resolve_data_dir(data_dir):
    """Ensure data_dir is the PARENT that contains a split sub-folder.

    MOFTransformer's Dataset(data_dir, split=...) looks for files in
    data_dir/<split>/ and requires split ∈ {train, test, val}.

    If the user points directly at the folder with .graphdata files:
      - go one level up (parent becomes data_dir)
      - if the folder name is already train/test/val → use as-is
      - otherwise → create a symlink  parent/test -> folder  and use 'test'

    Returns (data_dir, split_name, symlink_to_cleanup_or_None).
    """
    sample = [f for f in os.listdir(data_dir) if f.endswith(".graphdata")]
    if sample:
        parent = os.path.dirname(data_dir)
        basename = os.path.basename(data_dir)
        print(f"  NOTE: .graphdata files found directly in {data_dir}")
        print(f"        MOFTransformer expects data_dir/<split>/ layout.")

        if basename in _VALID_SPLITS:
            print(f"        Using parent dir: {parent}  (split='{basename}')")
            return parent, basename, None

        # Folder name not in {train, test, val} → symlink as 'test'
        link = os.path.join(parent, "test")
        if os.path.exists(link) or os.path.islink(link):
            # 'test' already exists — check if it points to the same place
            real_link = os.path.realpath(link)
            real_data = os.path.realpath(data_dir)
            if real_link == real_data:
                print(f"        Symlink {link} already points to {data_dir}")
                return parent, "test", None     # nothing to clean up
            else:
                raise RuntimeError(
                    f"{link} already exists but points to {real_link}, "
                    f"not {real_data}.  Rename/remove it and retry.")
        os.symlink(data_dir, link)
        print(f"        Created symlink: {link} -> {data_dir}")
        print(f"        Using parent dir: {parent}  (split='test')")
        return parent, "test", link

    # data_dir is the parent — check for a split sub-folder
    for candidate in ("test", "train", "val"):
        if os.path.isdir(os.path.join(data_dir, candidate)):
            return data_dir, candidate, None
    # Fallback
    return data_dir, "test", None


def discover_mof_ids(data_dir, split_subdir):
    """Find all unique MOF CIF-IDs that have .graphdata files."""
    search_dir = os.path.join(data_dir, split_subdir)
    ids = set()
    for fn in os.listdir(search_dir):
        if fn.endswith(".graphdata"):
            cid = fn[: -len(".graphdata")]
            ids.add(cid)
    return ids


def check_required_files(cid, data_dir, split_subdir):
    """Return True if all 3 required files exist (.graphdata, .griddata16, .grid)."""
    d = os.path.join(data_dir, split_subdir)
    for ext in (".graphdata", ".griddata16", ".grid"):
        if not os.path.exists(os.path.join(d, cid + ext)):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════
#  Build a temporary "test" split JSON covering everything
# ══════════════════════════════════════════════════════════════════════
def build_unified_json(mof_ids, bg_map, data_dir, split_name,
                       downstream="bandgaps_regression"):
    """Write a single {split}_{downstream}.json that lists ALL MOFs.

    MOFTransformer's Dataset reads labels from {split}_{downstream}.json
    in data_dir.  We create a temporary JSON so every MOF is loaded
    in one pass.

    Returns path to the temporary JSON.
    """
    labels = {}
    for cid in sorted(mof_ids):
        labels[cid] = bg_map.get(cid, 0.0)   # dummy 0 for unlabeled
    out_path = os.path.join(data_dir, f"{split_name}_{downstream}.json")
    # Back up if it already exists
    backup = None
    if os.path.exists(out_path):
        backup = out_path + ".bak_unified"
        shutil.copy2(out_path, backup)
    with open(out_path, "w") as fh:
        json.dump(labels, fh)
    print(f"  Wrote unified JSON ({len(labels)} MOFs) -> {out_path}")
    return out_path, backup


# ══════════════════════════════════════════════════════════════════════
#  Extraction
# ══════════════════════════════════════════════════════════════════════
def extract_all(model, data_dir, split_name, downstream, config,
                batch_size=1, num_workers=0, device="cuda"):
    """
    Extract 768-dim CLS embeddings for every MOF listed in
    data_dir/{split_name}_{downstream}.json.

    Uses batch_size=1 / num_workers=0 so we can log and skip individual
    failures (grid-shape mismatches).
    """
    model = model.to(device)
    model.eval()

    ds = Dataset(
        data_dir,
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
    print(f"  Dataset size: {total}")

    all_cid, all_emb = [], []
    n_done, n_skip = 0, 0
    skipped_ids = []
    t0 = time.time()
    last_report = t0

    def _progress(force=False):
        nonlocal last_report
        now = time.time()
        # report every 30 s, or when forced
        if not force and (now - last_report) < 30:
            return
        last_report = now
        elapsed = now - t0
        pct = n_done / total * 100 if total else 0
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (total - n_done) / rate if rate > 0 else 0
        eta_m, eta_s = divmod(int(remaining), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        el_m, el_s = divmod(int(elapsed), 60)
        el_h, el_m = divmod(el_m, 60)
        print(f"    [{n_done:>6}/{total}]  {pct:5.1f}%  "
              f"elapsed {el_h}h{el_m:02d}m{el_s:02d}s  "
              f"ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s  "
              f"({rate:.1f} MOF/s)  skipped {n_skip}",
              flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            try:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                output = model.infer(batch)
                cls_feats = output["cls_feats"]        # [B, 768]
                cids_batch = output["cif_id"]
                cls_np = cls_feats.cpu().numpy()
                for i, cid in enumerate(cids_batch):
                    all_cid.append(cid)
                    all_emb.append(cls_np[i])
                n_done += len(cids_batch)
            except RuntimeError as e:
                # Grid-shape mismatch — skip this MOF
                err = str(e)
                if "shape" in err or "invalid" in err or "size" in err:
                    n_skip += 1
                    skipped_ids.append(f"batch_{batch_idx}")
                    continue
                raise
            _progress()

    _progress(force=True)  # final line
    elapsed_total = time.time() - t0
    em, es = divmod(int(elapsed_total), 60)
    eh, em = divmod(em, 60)
    print(f"  Extraction complete: {n_done} embedded, {n_skip} skipped  "
          f"[{eh}h{em:02d}m{es:02d}s total]")
    if skipped_ids:
        print(f"  Skipped IDs (by batch): {skipped_ids[:20]}"
              f"{'...' if len(skipped_ids)>20 else ''}")

    embeddings = np.stack(all_emb, axis=0) if all_emb else np.empty((0, 768))
    return all_cid, embeddings


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(
        description="Extract pretrained PMTransformer embeddings for ALL MOFs in one run",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--data_dir", required=True,
                    help="Dir with .graphdata/.griddata16/.grid for ALL MOFs")
    pa.add_argument("--labeled_splits_dir", required=True,
                    help="Split dir with {train,val,test}_bandgaps_regression.json")
    pa.add_argument("--unlabeled_json", required=True,
                    help="Unlabeled test_bandgaps_regression.json (dummy labels)")
    pa.add_argument("--output_dir", default="./figures_output/pretrained_embeddings",
                    help="Output directory")
    pa.add_argument("--downstream", default="bandgaps_regression")
    pa.add_argument("--batch_size", type=int, default=1,
                    help="1 recommended (allows individual skip on failure)")
    pa.add_argument("--num_workers", type=int, default=0,
                    help="0 recommended for skip logging")
    pa.add_argument("--device", default=None,
                    help="Force device (default: auto cuda/cpu)")
    args = pa.parse_args()

    data_dir    = os.path.abspath(args.data_dir)
    splits_dir  = os.path.abspath(args.labeled_splits_dir)
    output_dir  = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  PRETRAINED EMBEDDING EXTRACTION -- ALL MOFs, SINGLE FORWARD PASS")
    print("=" * 70)

    # ── 0. Resolve data_dir layout ───────────────────────────────────
    data_dir, split_name, _symlink = resolve_data_dir(data_dir)
    print(f"  Resolved data_dir : {data_dir}")
    print(f"  Split sub-folder  : {split_name}/")

    # ── 1. Load labeled info ─────────────────────────────────────────
    print("\n[1/5] Loading labeled split info ...")
    bg_map, split_map = load_labeled_splits(splits_dir, args.downstream)
    labeled_cids = set(bg_map.keys())

    # ── 2. Load unlabeled IDs ────────────────────────────────────────
    print("\n[2/5] Loading unlabeled MOF IDs ...")
    unlabeled_cids = load_unlabeled_ids(args.unlabeled_json)
    print(f"  Unlabeled from JSON: {len(unlabeled_cids)}")

    # ── 3. Discover MOFs in data_dir ─────────────────────────────────
    print(f"\n[3/5] Discovering MOFs in {data_dir}/{split_name}/ ...")
    all_disk_ids = discover_mof_ids(data_dir, split_name)
    print(f"  .graphdata files found: {len(all_disk_ids)}")

    # Keep only MOFs with all required files
    valid_ids = set()
    n_incomplete = 0
    for cid in all_disk_ids:
        if check_required_files(cid, data_dir, split_name):
            valid_ids.add(cid)
        else:
            n_incomplete += 1
    if n_incomplete:
        print(f"  Skipped {n_incomplete} MOFs with incomplete files")
    print(f"  Valid MOFs (all 3 files present): {len(valid_ids)}")

    n_lab_found   = len(valid_ids & labeled_cids)
    n_unlab_found = len(valid_ids & unlabeled_cids)
    n_other       = len(valid_ids) - n_lab_found - n_unlab_found
    print(f"  Breakdown: {n_lab_found} labeled, {n_unlab_found} unlabeled, "
          f"{n_other} in neither JSON (will be treated as unlabeled)")

    # ── 4. Write unified test JSON ───────────────────────────────────
    print(f"\n[4/5] Building unified label JSON ...")
    json_path, backup_path = build_unified_json(valid_ids, bg_map, data_dir,
                                                 split_name, args.downstream)

    # ── 5. Extract embeddings ────────────────────────────────────────
    print(f"\n[5/5] Extracting embeddings (pretrained PMTransformer) ...")

    config = default_config_fn()
    config = json.loads(json.dumps(config))
    config["data_dir"]    = data_dir
    config["downstream"]  = args.downstream
    config["load_path"]   = "pmtransformer"
    config["loss_names"]  = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "regression": 1, "classification": 0,
    }
    config = get_valid_config(config)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = Module(config)
    model.eval()

    cif_ids, embeddings = extract_all(
        model, data_dir, split_name, args.downstream, config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Restore original test JSON if we backed it up ────────────────
    if backup_path and os.path.exists(backup_path):
        shutil.move(backup_path, json_path)
        print(f"  Restored original {json_path}")
    else:
        # Remove the one we created so we don't pollute the data dir
        if os.path.exists(json_path):
            os.remove(json_path)
            print(f"  Cleaned up temporary {json_path}")

    # ── Remove symlink we created (if any) ───────────────────────────
    if _symlink and os.path.islink(_symlink):
        os.unlink(_symlink)
        print(f"  Removed temporary symlink {_symlink}")

    # ── Assemble metadata arrays ─────────────────────────────────────
    n = len(cif_ids)
    bandgaps   = np.full(n, np.nan, dtype=np.float64)
    splits     = np.array(["unlabeled"] * n)
    is_labeled = np.zeros(n, dtype=bool)

    for i, cid in enumerate(cif_ids):
        if cid in bg_map:
            bandgaps[i]   = bg_map[cid]
            splits[i]     = split_map.get(cid, "labeled")
            is_labeled[i] = True

    n_lab  = int(is_labeled.sum())
    n_ulab = n - n_lab
    print(f"\n  Final: {n} MOFs ({n_lab} labeled, {n_ulab} unlabeled), "
          f"embedding dim = {embeddings.shape[1]}")

    # ── Save ─────────────────────────────────────────────────────────
    npz_path = os.path.join(output_dir, "all_embeddings.npz")
    np.savez_compressed(
        npz_path,
        cif_ids=np.array(cif_ids),
        embeddings=embeddings,
        bandgaps=bandgaps,
        splits=splits,
        is_labeled=is_labeled,
    )
    print(f"  Saved → {npz_path}")

    # ── Quick sanity check ───────────────────────────────────────────
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n  Embedding norm stats:  mean={norms.mean():.2f}  "
          f"std={norms.std():.2f}  min={norms.min():.2f}  max={norms.max():.2f}")
    if norms.min() < 1e-6:
        n_zero = (norms < 1e-6).sum()
        print(f"  WARNING: {n_zero} MOFs have near-zero embedding norms!")

    print(f"\n{'='*70}")
    print(f"  DONE. Output: {npz_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
