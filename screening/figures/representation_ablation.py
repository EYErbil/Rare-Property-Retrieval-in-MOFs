#!/usr/bin/env python3
"""Representation ablation: does the pretrained embedding actually help?

Holds the rare-class classifier protocol *fixed* and swaps only the input
representation. The protocol is an EXACT replica of the production classifier
in Tree_training/embedding_classifier.py ('smote_extra_trees'): combine
train+val (train block first, then val block, as production does), oversample
with the manual SMOTE recipe (5 synthetic positives per real positive,
interpolating to one of the 5 nearest positive neighbours; seed 42), then
ExtraTrees with 500 trees, min leaf 3, no class weight (seed 42). The label
rule is the production script's strict `gap < 1.0` (the run logs how many
labels sit exactly at 1.0 eV; expected 0, in which case '<' vs '<=' is
immaterial). Representations:

    1. frozen pretrained PMTransformer embedding   (the paper's choice)
    2. SOAP descriptors                            (geometry-only baseline)
    3. composition vector (element counts)         (OPTIONAL --qmof_csv;
                                                    excluded from the paper)

Each classifier is evaluated on the *same* held-out test partition with the
early-recognition metrics used elsewhere (EF@k, recall@k, AUC-PR, AUC-ROC).
Outputs a CSV (for the SI table) and a grouped-bar figure of EF@{25,50,100}.

This is the one analysis that directly tests the paper's central premise --
"pretrained representations carry the right information for band gaps." It needs
NO retraining of PMTransformer: the embeddings are frozen and already computed.

--------------------------------------------------------------------------------
Run:

  python representation_ablation.py \
      --embeddings_npz /path/to/embeddings/pmt_embeddings_qmof_all.npz \
      --soap_npz /path/to/soap_analysis/qmof_soap_descriptors.npz \
      --splits_dir /path/to/data/splits/strategy_d_farthest_point \
      --qmof_csv qmof.csv        # optional; composition baseline skipped if absent

Inputs (either --splits_dir OR --labels_csv provides the labels):
  --embeddings_npz  PMTransformer embeddings covering the labelled set
                    (keys 'cif_ids' + 'embeddings'; e.g. the Globus archive
                    pmt_embeddings_qmof_all.npz).
  --soap_npz        SOAP descriptors for the same structures. The QMOF SOAP NPZ
                    is ~(20k x 244k) ~ 20 GB, so by default it is STREAMED and
                    random-projected to --soap_k_proj dims (default 4,096; a
                    seeded Gaussian projection, distance-preserving) using the
                    same streaming loader as figure script 08. Set
                    --soap_k_proj 0 to load the full matrix instead (needs RAM).
  --splits_dir      directory with {train,val,test}_bandgaps_regression.json
                    (the Script-1 convention; ids may carry the _FSR suffix).
  --labels_csv      alternative: CSV with structure_id, split, true_hse_gap_ev.
  --qmof_csv        qmof.csv for the composition baseline (skipped if missing).
--------------------------------------------------------------------------------
Dependencies: numpy, scikit-learn, matplotlib, and _splits.py (same folder).
The default SOAP streaming path additionally needs stream_soap_random_project()
from the original chemical-space module (not shipped); pass --soap_k_proj 0 to
load the full SOAP matrix instead. imbalanced-learn is NOT needed: the
oversampler is the production manual-SMOTE replica.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _splits import (  # noqa: E402
    build_id_index, id_variants, load_all_splits, lookup_index, strip_id)

HIT_THRESHOLD_EV = 1.0
BUDGETS = (25, 50, 100)
N_TREES = 500
MIN_LEAF = 3
SEED = 42
N_SYNTH_PER_POS = 5  # production oversampling ratio (embedding_classifier.py)

ID_KEYS = ("cif_ids", "ids", "structure_ids", "names", "qmof_ids",
           "structure_id", "id")
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


# --------------------------------------------------------------------------- #
# Metrics (identical conventions to 02_compute_test_partition_ranking_metrics) #
# --------------------------------------------------------------------------- #
def _avg_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def auc_roc(y: np.ndarray, s: np.ndarray) -> float:
    npos, nneg = float(y.sum()), float((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _avg_ranks(s)
    return (r[y > 0.5].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def auc_pr(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / y.sum())


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    order = np.argsort(-s, kind="mergesort")[:k]
    return float(y[order].sum() / k)


def recall_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    if y.sum() == 0:
        return float("nan")
    k = min(k, len(y))
    order = np.argsort(-s, kind="mergesort")[:k]
    return float(y[order].sum() / y.sum())


def enrichment_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    prev = y.mean()
    return precision_at_k(y, s, k) / prev if prev > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Loading / feature construction                                              #
# --------------------------------------------------------------------------- #
def _norm_id(x) -> str:
    return str(x).strip()


def load_npz_features(path: Path, key: str,
                      cast32: bool = False) -> Tuple[np.ndarray, List[str]]:
    z = np.load(path, allow_pickle=True)
    if key not in z:
        # fall back to the first 2-D float array
        key = next(k for k in z.files if z[k].ndim == 2)
    # Production never casts, so keep the stored dtype by default; cast32 is
    # only for the optional full-SOAP path, where float32 halves the memory.
    X = np.asarray(z[key], dtype=np.float32) if cast32 else np.asarray(z[key])
    id_key = next((k for k in ID_KEYS if k in z.files), None)
    if id_key is None:
        raise SystemExit(f"{path}: no id array found (looked for {ID_KEYS}). "
                         "Add one, or tell me the key name.")
    ids = [_norm_id(v) for v in z[id_key]]
    if len(ids) != len(X):
        raise SystemExit(f"{path}: {len(ids)} ids vs {len(X)} feature rows.")
    return X, ids


def load_labels(path: Path) -> Dict[str, Tuple[str, float]]:
    out: Dict[str, Tuple[str, float]] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            sid = _norm_id(r.get("structure_id") or r.get("name") or "")
            split = (r.get("split") or "").strip().lower()
            try:
                gap = float(r.get("true_hse_gap_ev"))
            except (TypeError, ValueError):
                continue
            if sid and split:
                out[sid] = (split, gap)
    return out


def load_labels_from_splits(splits_dir: Path) -> Dict[str, Tuple[str, float]]:
    """{train,val,test}_bandgaps_regression.json -> sid -> (split, gap)."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    splits = load_all_splits(Path(splits_dir), logging.getLogger("ablation"))
    if not splits:
        raise SystemExit(f"no split JSONs found under {splits_dir}")
    out: Dict[str, Tuple[str, float]] = {}
    for split_name, entries in splits.items():
        for sid, gap in entries.items():
            out[strip_id(sid)] = (split_name, float(gap))
    return out


def parse_formula(formula: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for el, n in _ELEMENT_RE.findall(str(formula)):
        if not el:
            continue
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts


def align_rows(X: np.ndarray, ids: List[str], target: List[str]) -> np.ndarray:
    """Reorder feature rows so row i corresponds to ``target[i]``."""
    pos = {sid: i for i, sid in enumerate(ids)}
    return X[[pos[s] for s in target]]


def composition_features(ids: List[str],
                         qmof_csv: Path) -> Tuple[np.ndarray, List[str]]:
    by_id: Dict[str, str] = {}
    with open(qmof_csv) as fh:
        for r in csv.DictReader(fh):
            f = r.get("info.formula") or r.get("info.formula_reduced") or ""
            for key in ("name", "qmof_id"):
                if r.get(key):
                    for v in id_variants(_norm_id(r[key])):
                        by_id.setdefault(v, f)
    parsed = [parse_formula(next((by_id[v] for v in id_variants(i)
                                  if v in by_id), ""))
              for i in ids]
    n_matched = sum(1 for p in parsed if p)
    elements = sorted({e for p in parsed for e in p})
    X = np.zeros((len(ids), len(elements)), float)
    idx = {e: j for j, e in enumerate(elements)}
    for i, p in enumerate(parsed):
        for e, n in p.items():
            X[i, idx[e]] = n
    return X, elements, n_matched


# --------------------------------------------------------------------------- #
# Train + evaluate one representation                                         #
# --------------------------------------------------------------------------- #
def smote_manual(X: np.ndarray, y: np.ndarray,
                 n_synthetic_per_pos: int = N_SYNTH_PER_POS,
                 k_neighbors: int = 5,
                 random_state: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Verbatim replica of Tree_training/embedding_classifier.py::smote_manual
    (the production oversampler): each positive spawns n_synthetic_per_pos
    synthetic points by interpolating towards one of its k nearest positive
    neighbours (lambda ~ U[0.1, 0.9])."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(random_state)
    pos_mask = y == 1
    X_pos = X[pos_mask]
    n_pos = len(X_pos)
    if n_pos < 2:
        print("  SMOTE: fewer than 2 positives, cannot interpolate")
        return X, y
    k = min(k_neighbors, n_pos - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_pos)
    _, indices = nn.kneighbors(X_pos)
    synthetic = []
    for i in range(n_pos):
        neighbors = indices[i, 1:]
        for _ in range(n_synthetic_per_pos):
            j = rng.choice(neighbors)
            lam = rng.uniform(0.1, 0.9)
            synthetic.append(X_pos[i] + lam * (X_pos[j] - X_pos[i]))
    synthetic = np.array(synthetic)
    X_aug = np.vstack([X, synthetic])
    y_aug = np.concatenate([y, np.ones(len(synthetic), dtype=y.dtype)])
    print(f"  SMOTE: {n_pos} pos -> {n_pos + len(synthetic)} pos "
          f"(+{len(synthetic)} synthetic, {n_synthetic_per_pos}x)")
    return X_aug, y_aug


def evaluate_representation(name: str, X: np.ndarray, ids: List[str],
                            labels: Dict[str, Tuple[str, float]]) -> Dict:
    from sklearn.ensemble import ExtraTreesClassifier

    keep = [i for i, sid in enumerate(ids) if sid in labels]
    X, ids = X[keep], [ids[i] for i in keep]
    split = np.array([labels[i][0] for i in ids])
    gap = np.array([labels[i][1] for i in ids])
    # Strict '<', exactly as in the production script. Log the boundary case:
    # if no label equals the threshold, '<' and '<=' give identical classes.
    n_at_thr = int((gap == HIT_THRESHOLD_EV).sum())
    if n_at_thr:
        print(f"  WARNING [{name}]: {n_at_thr} label(s) equal exactly "
              f"{HIT_THRESHOLD_EV} eV -- '<' (production) and '<=' differ!")
    y = (gap < HIT_THRESHOLD_EV).astype(int)

    is_train = split == "train"
    is_val = np.isin(split, ("val", "validation"))
    te = split == "test"
    # Production row order: train block first, then val block.
    Xtr = np.vstack([X[is_train], X[is_val]])
    ytr = np.concatenate([y[is_train], y[is_val]])
    Xte, yte = X[te], y[te]
    n_pos = int(ytr.sum())
    if n_pos < 2 or yte.sum() == 0:
        raise SystemExit(f"[{name}] too few positives (train={n_pos}, "
                         f"test={int(yte.sum())}).")

    Xrs, yrs = smote_manual(Xtr, ytr)
    clf = ExtraTreesClassifier(n_estimators=N_TREES, max_depth=None,
                               min_samples_leaf=MIN_LEAF,
                               random_state=SEED, n_jobs=-1)
    clf.fit(Xrs, yrs)
    score = clf.predict_proba(Xte)[:, 1]

    row = {"representation": name, "n_features": X.shape[1],
           "n_train": len(ytr), "n_train_pos": n_pos,
           "n_test": int(te.sum()), "n_test_pos": int(yte.sum()),
           "auc_roc": auc_roc(yte, score), "auc_pr": auc_pr(yte, score)}
    for k in BUDGETS:
        row[f"precision_at_{k}"] = precision_at_k(yte, score, k)
        row[f"recall_at_{k}"] = recall_at_k(yte, score, k)
        row[f"ef_at_{k}"] = enrichment_at_k(yte, score, k)
    print(f"[{name:12s}] feats={X.shape[1]:5d}  "
          + "  ".join(f"EF@{k}={row[f'ef_at_{k}']:.0f}" for k in BUDGETS)
          + f"  AUC-PR={row['auc_pr']:.3f}")
    return row


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #
def make_figure(rows: List[Dict], out_stem: Path) -> None:
    import matplotlib.pyplot as plt
    base_colours = {"PMTransformer": "#4477AA",
                    "SOAP": "#EE6677",
                    "Composition": "#BBBBBB"}

    def colour_for(label: str) -> str:
        return next((c for k, c in base_colours.items()
                     if label.startswith(k)), "#999999")
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    x = np.arange(len(BUDGETS))
    w = 0.26
    top = max((row[f"ef_at_{k}"] for row in rows for k in BUDGETS), default=1.0) or 1.0
    for i, row in enumerate(rows):
        vals = [row[f"ef_at_{k}"] for k in BUDGETS]
        off = x + (i - (len(rows) - 1) / 2) * w
        ax.bar(off, vals, width=w, color=colour_for(row["representation"]),
               edgecolor="white", linewidth=0.5, label=row["representation"])
        for xpos, v in zip(off, vals):
            ax.text(xpos, v + 0.015 * top, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=6.5)
    ax.axhline(1.0, ls=(0, (4, 2)), lw=1.0, color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in BUDGETS])
    ax.set_xlabel("Screening budget $k$")
    ax.set_ylabel(r"Enrichment over random ($\times$)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.0, frameon=False, title="Classifier input")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print("wrote", out_stem.with_suffix(".pdf"))


def main() -> int:
    code = Path(__file__).resolve().parent
    pa = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--embeddings_npz", type=Path, required=True)
    pa.add_argument("--embeddings_key", default="embeddings")
    pa.add_argument("--soap_npz", type=Path, required=True)
    pa.add_argument("--soap_key", default="soap_descriptors")
    pa.add_argument("--labels_csv", type=Path, default=None)
    pa.add_argument("--splits_dir", type=Path, default=None,
                    help="dir with {train,val,test}_bandgaps_regression.json")
    pa.add_argument("--soap_k_proj", type=int, default=4096,
                    help="stream the SOAP NPZ and random-project to this many "
                         "dims (0 = load the full matrix into RAM)")
    pa.add_argument("--qmof_csv", type=Path, default=None,
                    help="OPTIONAL extra composition baseline (element counts "
                         "from qmof.csv). Excluded from the manuscript; only "
                         "computed when this flag is passed explicitly.")
    pa.add_argument("--output_dir", type=Path, default=code.parent)
    args = pa.parse_args()

    if bool(args.labels_csv) == bool(args.splits_dir):
        raise SystemExit("provide exactly one of --labels_csv / --splits_dir")
    labels = (load_labels(args.labels_csv) if args.labels_csv
              else load_labels_from_splits(args.splits_dir))
    print(f"labels: {len(labels)} structures "
          f"({sum(g <= HIT_THRESHOLD_EV for _, g in labels.values())} positive)")

    # Evaluate every representation on the SAME structures, so differences are
    # due to the representation and not to which structures each one happens to
    # cover (a fair, like-for-like ablation). IDs are joined with the Script-1
    # variant logic (handles the _FSR suffix / .cif extension conventions).
    Xe_all, ide = load_npz_features(args.embeddings_npz, args.embeddings_key)
    emb_idx = build_id_index(ide)
    cand = [sid for sid in sorted(labels)
            if lookup_index(sid, emb_idx) is not None]
    print(f"labelled structures found in the embedding NPZ: {len(cand)}")

    if args.soap_k_proj > 0:
        # Stream the (huge) SOAP NPZ and random-project on the fly -- same
        # memory-bounded loader used by figure script 08.
        import logging
        from importlib import import_module
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        try:
            chem = import_module("08_make_fig6_chemical_space")
        except ModuleNotFoundError:
            raise SystemExit(
                "The SOAP streaming path needs stream_soap_random_project() "
                "from the original analysis module "
                "'08_make_fig6_chemical_space.py', which is not shipped. "
                "Place that file next to this script, or rerun with "
                "--soap_k_proj 0 to load the full SOAP matrix (needs RAM).")
        Xs_all, ok = chem.stream_soap_random_project(
            args.soap_npz, cand, args.soap_k_proj, SEED,
            logging.getLogger("ablation"))
        if Xs_all is None or not ok.any():
            raise SystemExit("SOAP streaming matched no ids -- check the NPZ.")
        common = [sid for sid, o in zip(cand, ok) if o]
        Xs = Xs_all[ok]
        soap_label = f"SOAP (random proj. {args.soap_k_proj}d)"
    else:
        Xs_full, ids = load_npz_features(args.soap_npz, args.soap_key,
                                         cast32=True)
        soap_idx = build_id_index(ids)
        common = [sid for sid in cand
                  if lookup_index(sid, soap_idx) is not None]
        Xs = Xs_full[[lookup_index(s, soap_idx) for s in common]]
        soap_label = "SOAP"

    if len(common) < 50:
        raise SystemExit(f"only {len(common)} structures shared across "
                         "embeddings, SOAP, and labels -- check id alignment.")
    print(f"common structures across all representations: {len(common)}")
    Xe = Xe_all[[lookup_index(s, emb_idx) for s in common]]

    rows = [
        evaluate_representation("PMTransformer embedding", Xe, common, labels),
        evaluate_representation(soap_label, Xs, common, labels),
    ]
    if args.qmof_csv is not None and args.qmof_csv.exists():
        Xc, elements, n_matched = composition_features(common, args.qmof_csv)
        if elements and n_matched >= 0.5 * len(common):
            rows.append(
                evaluate_representation("Composition", Xc, common, labels))
        else:
            print(f"NOTE: composition join matched only {n_matched}/"
                  f"{len(common)} ids in {args.qmof_csv} -- composition "
                  "baseline skipped (id convention mismatch?).")
    elif args.qmof_csv is not None:
        print(f"NOTE: {args.qmof_csv} not found -- composition baseline "
              "skipped (pass a valid --qmof_csv to compute it).")

    data_dir = args.output_dir / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    cols = (["representation", "n_features", "n_train", "n_train_pos",
             "n_test", "n_test_pos", "auc_roc", "auc_pr"]
            + [f"{m}_at_{k}" for k in BUDGETS for m in ("precision", "recall", "ef")])
    with open(data_dir / "representation_ablation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print("wrote", data_dir / "representation_ablation.csv")

    make_figure(rows, args.output_dir / "figures" / "main" / "figure_representation_ablation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
