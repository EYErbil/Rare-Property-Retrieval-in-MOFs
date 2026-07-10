#!/usr/bin/env python3
"""
Model Comparison — NN vs ML Top-K on UMAP embedding space.

Loads unlabeled embeddings (.npz), ranks each model's predictions, and
visualises where each model's top-K candidates sit relative to the full
~10 K structure cloud.  No re-extraction needed — just name-matching.

Outputs (in --output_dir):
  discovery_umap_per_model_top{K}.png   — grid of subplots, one per model
  discovery_nn_vs_ml_top{K}.png         — NN-union vs ML-union consensus
  discovery_nn_vs_ml_density_top{K}.png — side-by-side vote-count density
  discovery_jaccard_top{K}.png          — pairwise Jaccard heatmap
  model_comparison_report_top{K}.txt    — detailed text report
  umap_coords_discovery.npz             — cached 2-D UMAP coords

Usage:
  python plot_model_comparison.py                       # defaults
  python plot_model_comparison.py --top_k 50            # top-50
  python plot_model_comparison.py --all_ml              # all 31 ML classifiers
  python plot_model_comparison.py --ml_methods smote_extra_trees random_forest
"""

import os
import sys
import csv
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EMB = SCRIPT_DIR / "embedding_analysis" / "unlabeled_embeddings.npz"
DEFAULT_EXPERIMENTS = SCRIPT_DIR / "experiments"
DEFAULT_CLASSIFIERS = (
    SCRIPT_DIR / "embedding_classifiers" / "strategy_d_farthest_point"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "model_comparison"


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_embeddings(npz_path):
    """Load unlabeled embeddings .npz → (cif_ids, embeddings)."""
    data = np.load(npz_path, allow_pickle=True)
    cif_ids = data["cif_ids"]
    embeddings = data["embeddings"]
    if cif_ids.dtype.kind == "S" or (
        len(cif_ids) > 0 and isinstance(cif_ids[0], bytes)
    ):
        cif_ids = np.array(
            [c.decode() if isinstance(c, bytes) else c for c in cif_ids]
        )
    return cif_ids, embeddings


def load_nn_models(experiments_dir, top_k=25):
    """Load NN model predictions from experiments/."""
    models = {}
    exp_dir = Path(experiments_dir)
    if not exp_dir.is_dir():
        return models

    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir():
            continue

        ranked_csv = sub / "inference_ranked.csv"
        pred_csv = sub / "inference_predictions.csv"
        top25_file = sub / "top25_for_DFT.txt"

        all_ranked = []
        if ranked_csv.is_file():
            with open(ranked_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("cif_id", "").strip()
                    try:
                        score = float(row.get("score", 0))
                    except ValueError:
                        continue
                    if cid:
                        all_ranked.append((cid, score))
        elif pred_csv.is_file():
            rows = []
            with open(pred_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("cif_id", "").strip()
                    try:
                        score = float(row.get("score", 0))
                    except ValueError:
                        continue
                    if cid:
                        rows.append((cid, score))
            # NN regression: lower score = better
            all_ranked = sorted(rows, key=lambda x: x[1])

        if not all_ranked and top25_file.is_file():
            with open(top25_file, "r", encoding="utf-8") as f:
                top_cids = [line.strip() for line in f if line.strip()]
            all_ranked = [(c, float(i)) for i, c in enumerate(top_cids)]

        if all_ranked:
            # Short key: exp364, exp370, exp371
            name = sub.name
            short = name.split("_")[0] if name.startswith("exp") else name
            models[short] = {
                "top_k": [c for c, _ in all_ranked[:top_k]],
                "all_ranked": all_ranked,
                "type": "NN",
                "full_name": name,
            }
    return models


def load_ml_models(classifiers_dir, top_k=25, methods=None):
    """Load ML model predictions from embedding_classifiers/."""
    models = {}
    clf_dir = Path(classifiers_dir)
    if not clf_dir.is_dir():
        return models

    for sub in sorted(clf_dir.iterdir()):
        if not sub.is_dir():
            continue
        if methods and sub.name not in methods:
            continue

        csv_path = sub / "test_predictions.csv"
        if not csv_path.is_file():
            continue

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("cif_id", "").strip()
                try:
                    score = float(row.get("score", 0))
                except ValueError:
                    continue
                if cid:
                    rows.append((cid, score))

        if not rows:
            continue

        # Multiclass: higher score = more positive
        all_ranked = sorted(rows, key=lambda x: x[1], reverse=True)
        models[sub.name] = {
            "top_k": [c for c, _ in all_ranked[:top_k]],
            "all_ranked": all_ranked,
            "type": "ML",
            "full_name": sub.name,
        }
    return models


# ═══════════════════════════════════════════════════════════════════════
# UMAP
# ═══════════════════════════════════════════════════════════════════════

def compute_or_load_umap(embeddings, cache_path,
                         n_neighbors=30, min_dist=0.3):
    """Compute UMAP or load from cache."""
    cache = Path(cache_path)
    if cache.is_file():
        data = np.load(cache)
        if ("umap_coords" in data
                and data["umap_coords"].shape[0] == embeddings.shape[0]):
            print(f"  Loaded cached UMAP coords from {cache}")
            return data["umap_coords"]

    try:
        from umap import UMAP
    except ImportError:
        print("ERROR: umap-learn not installed. pip install umap-learn")
        sys.exit(1)

    print(f"  Computing UMAP on {embeddings.shape[0]} × "
          f"{embeddings.shape[1]} embeddings ...")
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                   metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)
    np.savez_compressed(cache, umap_coords=coords)
    print(f"  Saved UMAP coords → {cache}")
    return coords


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY NAMES
# ═══════════════════════════════════════════════════════════════════════

_DISPLAY_NAMES = {
    "exp364": "exp364 (NN s1)",
    "exp370": "exp370 (NN s2)",
    "exp371": "exp371 (NN s3)",
    "smote_extra_trees": "SMOTE Extra-Trees (ML)",
    "smote_random_forest": "SMOTE Random-Forest (ML)",
    "extra_trees": "Extra-Trees (ML)",
    "random_forest": "Random-Forest (ML)",
    "gradient_boosting": "Grad-Boost (ML)",
    "logistic_regression": "LogReg (ML)",
    "adaboost": "AdaBoost (ML)",
    "xgboost_regression": "XGBoost Reg (ML)",
    "svm_rbf": "SVM-RBF (ML)",
    "gaussian_nb": "GaussNB (ML)",
    "knn_classifier": "kNN (ML)",
    "mlp_classifier": "MLP (ML)",
    "lda": "LDA (ML)",
    "isolation_forest": "Iso-Forest (ML)",
    "ridge_regression": "Ridge Reg (ML)",
    "lasso_regression": "Lasso Reg (ML)",
    "rf_regression": "RF Reg (ML)",
    "label_spreading": "Label-Spread (ML)",
    "pca32_logreg": "PCA32+LogReg (ML)",
    "two_stage_knn_et": "2Stage kNN+ET (ML)",
    "smote_gradient_boosting": "SMOTE Grad-Boost (ML)",
    "fs100_extra_trees": "FS100 Extra-Trees (ML)",
    "fs100_random_forest": "FS100 Random-Forest (ML)",
    "fs100_knn": "FS100 kNN (ML)",
    "fs100_logreg": "FS100 LogReg (ML)",
    "mahalanobis": "Mahalanobis (ML)",
}


def display_name(key, model_type=""):
    if key in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[key]
    tag = f" ({model_type})" if model_type else ""
    return key.replace("_", " ").title() + tag


# ═══════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def compute_overlap_stats(models, top_k):
    names = sorted(models.keys())
    n = len(names)
    top_sets = {nm: set(models[nm]["top_k"][:top_k]) for nm in names}

    # Pairwise Jaccard
    jm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = top_sets[names[i]], top_sets[names[j]]
            inter = len(a & b)
            union = len(a | b)
            jm[i, j] = inter / union if union else 1.0

    nn_names = [nm for nm in names if models[nm]["type"] == "NN"]
    ml_names = [nm for nm in names if models[nm]["type"] == "ML"]

    nn_union = set().union(*(top_sets[n] for n in nn_names)) if nn_names else set()
    ml_union = set().union(*(top_sets[n] for n in ml_names)) if ml_names else set()
    nn_inter = (set.intersection(*(top_sets[n] for n in nn_names))
                if nn_names else set())
    ml_inter = (set.intersection(*(top_sets[n] for n in ml_names))
                if ml_names else set())

    return {
        "names": names,
        "jaccard_mat": jm,
        "nn_union": nn_union, "ml_union": ml_union,
        "nn_intersect": nn_inter, "ml_intersect": ml_inter,
        "cross_overlap": nn_union & ml_union,
        "top_sets": top_sets,
    }


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def _init_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_per_model_grid(coords, cif_ids, models, top_k, output_path):
    """Grid of subplots — one per model showing its top-K on UMAP."""
    plt = _init_mpl()
    cid_to_idx = {c: i for i, c in enumerate(cif_ids)}
    names = sorted(models.keys(),
                   key=lambda n: (0 if models[n]["type"] == "NN" else 1, n))
    n = len(names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 6 * nrows), squeeze=False)

    for idx, name in enumerate(names):
        info = models[name]
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.scatter(coords[:, 0], coords[:, 1],
                   c="lightgray", s=1, alpha=0.2, rasterized=True)

        top_cids = info["top_k"][:top_k]
        top_idx = [cid_to_idx[c] for c in top_cids if c in cid_to_idx]

        colour = "crimson" if info["type"] == "NN" else "royalblue"
        marker = "*" if info["type"] == "NN" else "^"
        if top_idx:
            ax.scatter(coords[top_idx, 0], coords[top_idx, 1],
                       c=colour, s=140, marker=marker,
                       edgecolors="black", linewidths=0.5,
                       zorder=5, alpha=0.9)
            # Label the #1 candidate
            ax.annotate(top_cids[0],
                        xy=(coords[top_idx[0], 0], coords[top_idx[0], 1]),
                        fontsize=7, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")

        disp = display_name(name, info["type"])
        ax.set_title(f"{disp}\nTop-{top_k}: {len(top_idx)}/{len(top_cids)}",
                     fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Discovery UMAP — Per-Model Top-{top_k} Candidates",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_nn_vs_ml_consensus(coords, cif_ids, nn_models, ml_models,
                            top_k, output_path):
    """Single plot: NN-union vs ML-union top-K with overlap."""
    plt = _init_mpl()
    cid_to_idx = {c: i for i, c in enumerate(cif_ids)}

    nn_union = set()
    for info in nn_models.values():
        nn_union |= set(info["top_k"][:top_k])
    ml_union = set()
    for info in ml_models.values():
        ml_union |= set(info["top_k"][:top_k])

    nn_only = nn_union - ml_union
    ml_only = ml_union - nn_union
    both = nn_union & ml_union

    def to_idx(s):
        return [cid_to_idx[c] for c in s if c in cid_to_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1],
               c="lightgray", s=1, alpha=0.2, rasterized=True)

    idx_nn = to_idx(nn_only)
    if idx_nn:
        ax.scatter(coords[idx_nn, 0], coords[idx_nn, 1],
                   c="crimson", s=150, marker="*",
                   edgecolors="darkred", linewidths=0.5, zorder=5,
                   alpha=0.9, label=f"NN-only ({len(idx_nn)})")
    idx_ml = to_idx(ml_only)
    if idx_ml:
        ax.scatter(coords[idx_ml, 0], coords[idx_ml, 1],
                   c="royalblue", s=120, marker="^",
                   edgecolors="navy", linewidths=0.5, zorder=5,
                   alpha=0.9, label=f"ML-only ({len(idx_ml)})")
    idx_both = to_idx(both)
    if idx_both:
        ax.scatter(coords[idx_both, 0], coords[idx_both, 1],
                   c="limegreen", s=180, marker="D",
                   edgecolors="darkgreen", linewidths=0.8, zorder=6,
                   alpha=0.95, label=f"Both NN+ML ({len(idx_both)})")

    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax.set_title(
        f"Discovery UMAP — NN vs ML Top-{top_k} Consensus\n"
        f"NN union: {len(nn_union)} | ML union: {len(ml_union)} "
        f"| Overlap: {len(both)}",
        fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_nn_vs_ml_sidebyside(coords, cif_ids, nn_models, ml_models,
                             top_k, output_path):
    """Side-by-side panels coloured by vote count per type."""
    plt = _init_mpl()
    from matplotlib.colors import Normalize

    cid_to_idx = {c: i for i, c in enumerate(cif_ids)}

    nn_votes = defaultdict(int)
    ml_votes = defaultdict(int)
    for info in nn_models.values():
        for c in info["top_k"][:top_k]:
            nn_votes[c] += 1
    for info in ml_models.values():
        for c in info["top_k"][:top_k]:
            ml_votes[c] += 1

    n_nn = len(nn_models)
    n_ml = len(ml_models)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for ax, votes, n_total, title, cmap in [
        (ax1, nn_votes, n_nn, f"NN Models (n={n_nn})", "Reds"),
        (ax2, ml_votes, n_ml, f"ML Models (n={n_ml})", "Blues"),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1],
                   c="lightgray", s=1, alpha=0.15, rasterized=True)
        items = [(c, v) for c, v in votes.items() if c in cid_to_idx]
        if items:
            items.sort(key=lambda x: x[1])
            idxs = [cid_to_idx[c] for c, _ in items]
            vals = [v for _, v in items]
            norm = Normalize(vmin=1, vmax=max(n_total, 1))
            sc = ax.scatter(coords[idxs, 0], coords[idxs, 1],
                            c=vals, cmap=cmap, norm=norm,
                            s=80, edgecolors="black", linewidths=0.3,
                            zorder=5, alpha=0.9)
            plt.colorbar(sc, ax=ax,
                         label=f"# models with CIF in top-{top_k}",
                         shrink=0.8)
        ax.set_title(f"{title} — Top-{top_k} candidates", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Discovery UMAP — NN vs ML Vote Density (Top-{top_k})",
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_jaccard_heatmap(stats, top_k, models, output_path):
    plt = _init_mpl()
    names = stats["names"]
    jm = stats["jaccard_mat"]
    n = len(names)
    labels = [display_name(nm, models[nm]["type"]) for nm in names]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(6, n * 0.8)))
    im = ax.imshow(jm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{jm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if jm[i, j] > 0.5 else "black")

    ax.set_title(f"Pairwise Jaccard Similarity — Top-{top_k}", fontsize=13)
    plt.colorbar(im, ax=ax, label="Jaccard", shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════

def write_report(models, stats, top_k, output_path):
    L = []
    L.append("=" * 70)
    L.append(f"Model Comparison Report  (top-{top_k})")
    L.append("=" * 70)
    L.append("")

    nn_names = [n for n in stats["names"] if models[n]["type"] == "NN"]
    ml_names = [n for n in stats["names"] if models[n]["type"] == "ML"]
    L.append(f"Models: {len(nn_names)} NN + {len(ml_names)} ML "
             f"= {len(stats['names'])} total")
    L.append("")

    # ── Per-model top-K ───────────────────────────────────────────────
    L.append("-" * 70)
    L.append("PER-MODEL TOP-K LISTS")
    L.append("-" * 70)
    for name in stats["names"]:
        disp = display_name(name, models[name]["type"])
        top = models[name]["top_k"][:top_k]
        L.append(f"\n{disp}:")
        for i, c in enumerate(top, 1):
            L.append(f"  {i:3d}. {c}")
    L.append("")

    # ── Cross-type analysis ───────────────────────────────────────────
    L.append("-" * 70)
    L.append("CROSS-TYPE ANALYSIS  (NN vs ML)")
    L.append("-" * 70)
    L.append(f"NN union   (in ANY  NN top-{top_k}):  "
             f"{len(stats['nn_union'])} unique CIFs")
    L.append(f"NN intersect (ALL NN top-{top_k}):    "
             f"{len(stats['nn_intersect'])} CIFs")
    L.append(f"ML union   (in ANY  ML top-{top_k}):  "
             f"{len(stats['ml_union'])} unique CIFs")
    L.append(f"ML intersect (ALL ML top-{top_k}):    "
             f"{len(stats['ml_intersect'])} CIFs")
    L.append(f"Cross-type overlap (NN∩ML unions):     "
             f"{len(stats['cross_overlap'])} CIFs")
    L.append("")

    if stats["cross_overlap"]:
        L.append(f"CIFs in BOTH NN and ML top-{top_k} unions:")
        for c in sorted(stats["cross_overlap"]):
            L.append(f"  * {c}")
    else:
        L.append(f"WARNING: ZERO overlap between NN and ML at top-{top_k}!")
        L.append("  The two model families completely disagree on candidates.")
    L.append("")

    if stats["nn_intersect"]:
        L.append(f"CIFs agreed by ALL {len(nn_names)} NN models:")
        for c in sorted(stats["nn_intersect"]):
            L.append(f"  * {c}")
        L.append("")
    if stats["ml_intersect"]:
        L.append(f"CIFs agreed by ALL {len(ml_names)} ML models:")
        for c in sorted(stats["ml_intersect"]):
            L.append(f"  * {c}")
        L.append("")

    nn_only = stats["nn_union"] - stats["ml_union"]
    ml_only = stats["ml_union"] - stats["nn_union"]
    L.append(f"NN-only CIFs ({len(nn_only)}):")
    for c in sorted(nn_only):
        L.append(f"  - {c}")
    L.append("")
    L.append(f"ML-only CIFs ({len(ml_only)}):")
    for c in sorted(ml_only):
        L.append(f"  - {c}")
    L.append("")

    # ── Pairwise Jaccard ──────────────────────────────────────────────
    L.append("-" * 70)
    L.append("PAIRWISE JACCARD SIMILARITY")
    L.append("-" * 70)
    names = stats["names"]
    jm = stats["jaccard_mat"]
    short = [display_name(n, models[n]["type"])[:22] for n in names]
    header = f"{'':>24}" + "".join(f"{s:>24}" for s in short)
    L.append(header)
    for i, s in enumerate(short):
        row = f"{s:>24}" + "".join(f"{jm[i, j]:>24.3f}"
                                   for j in range(len(names)))
        L.append(row)
    L.append("")

    nn_nn, ml_ml, nn_ml = [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ti, tj = models[names[i]]["type"], models[names[j]]["type"]
            v = jm[i, j]
            if ti == "NN" and tj == "NN":
                nn_nn.append(v)
            elif ti == "ML" and tj == "ML":
                ml_ml.append(v)
            else:
                nn_ml.append(v)

    L.append("Average Jaccard by type pairing:")
    if nn_nn:
        L.append(f"  NN <-> NN : {np.mean(nn_nn):.3f}  "
                 f"(n={len(nn_nn)} pairs)")
    if ml_ml:
        L.append(f"  ML <-> ML : {np.mean(ml_ml):.3f}  "
                 f"(n={len(ml_ml)} pairs)")
    if nn_ml:
        L.append(f"  NN <-> ML : {np.mean(nn_ml):.3f}  "
                 f"(n={len(nn_ml)} pairs)")
    L.append("")

    # ── Recommendation ────────────────────────────────────────────────
    L.append("-" * 70)
    L.append("RECOMMENDATIONS")
    L.append("-" * 70)
    cross_j = np.mean(nn_ml) if nn_ml else 0
    if cross_j < 0.05:
        L.append("CRITICAL: Near-zero NN<->ML Jaccard.")
        L.append("  -> Standard ensemble will be dominated by NN (3 vs 2).")
        L.append("  -> Use TYPE-BALANCED ensemble (equal weight per type).")
        L.append("  -> Investigate WHY the families disagree (embedding "
                 "space regions?).")
    elif cross_j < 0.15:
        L.append("LOW NN<->ML agreement.")
        L.append("  -> Type-balanced ensemble recommended.")
        L.append("  -> Cross-type overlap CIFs are high-confidence picks.")
    elif cross_j < 0.3:
        L.append("MODERATE NN<->ML agreement.")
        L.append("  -> Standard ensemble acceptable; type-balanced adds "
                 "diversity.")
    else:
        L.append("GOOD NN<->ML agreement — standard ensemble is fine.")
    L.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  Saved {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Discovery NN vs ML top-K investigation on UMAP")
    parser.add_argument("--embeddings_npz", type=str,
                        default=str(DEFAULT_EMB))
    parser.add_argument("--experiments_dir", type=str,
                        default=str(DEFAULT_EXPERIMENTS))
    parser.add_argument("--classifiers_dir", type=str,
                        default=str(DEFAULT_CLASSIFIERS))
    parser.add_argument("--output_dir", type=str,
                        default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--ml_methods", type=str, nargs="*", default=None,
                        help="ML methods to include (default: the 2 "
                             "SMOTE variants used in the ensemble)")
    parser.add_argument("--all_ml", action="store_true",
                        help="Include ALL available ML classifiers")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ml_methods = args.ml_methods
    if ml_methods is None and not args.all_ml:
        ml_methods = ["smote_extra_trees", "smote_random_forest"]

    # ── Load ──────────────────────────────────────────────────────────
    print(f"Loading embeddings from {args.embeddings_npz} ...")
    cif_ids, embeddings = load_embeddings(args.embeddings_npz)
    print(f"  {len(cif_ids)} structures, {embeddings.shape[1]}-dim")

    print("Loading model predictions ...")
    nn_models = load_nn_models(args.experiments_dir, top_k=args.top_k)
    ml_models = load_ml_models(args.classifiers_dir, top_k=args.top_k,
                               methods=ml_methods)
    print(f"  NN models: {sorted(nn_models.keys())}")
    print(f"  ML models: {sorted(ml_models.keys())}")

    if not nn_models and not ml_models:
        print("ERROR: No model predictions found.")
        return 1

    all_models = {**nn_models, **ml_models}

    # ── UMAP ──────────────────────────────────────────────────────────
    umap_cache = output_dir / "umap_coords_discovery.npz"
    coords = compute_or_load_umap(embeddings, umap_cache)

    # ── Stats ─────────────────────────────────────────────────────────
    print("Computing overlap statistics ...")
    stats = compute_overlap_stats(all_models, args.top_k)

    # ── Plots ─────────────────────────────────────────────────────────
    k = args.top_k
    print("Generating plots ...")

    plot_per_model_grid(
        coords, cif_ids, all_models, k,
        output_dir / f"discovery_umap_per_model_top{k}.png")

    if nn_models and ml_models:
        plot_nn_vs_ml_consensus(
            coords, cif_ids, nn_models, ml_models, k,
            output_dir / f"discovery_nn_vs_ml_top{k}.png")
        plot_nn_vs_ml_sidebyside(
            coords, cif_ids, nn_models, ml_models, k,
            output_dir / f"discovery_nn_vs_ml_density_top{k}.png")

    plot_jaccard_heatmap(
        stats, k, all_models,
        output_dir / f"discovery_jaccard_top{k}.png")

    write_report(
        all_models, stats, k,
        output_dir / f"model_comparison_report_top{k}.txt")

    # ── Quick summary ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    print(f"NN union: {len(stats['nn_union'])} | "
          f"ML union: {len(stats['ml_union'])} | "
          f"Overlap: {len(stats['cross_overlap'])}")
    if stats["cross_overlap"]:
        print(f"Cross-type CIFs: "
              f"{', '.join(sorted(stats['cross_overlap']))}")
    else:
        print(f"  ZERO cross-type overlap at top-{k}!")

    nn_ml_j = []
    for i in range(len(stats["names"])):
        for j in range(i + 1, len(stats["names"])):
            if (all_models[stats["names"][i]]["type"]
                    != all_models[stats["names"][j]]["type"]):
                nn_ml_j.append(stats["jaccard_mat"][i, j])
    if nn_ml_j:
        print(f"Average NN<->ML Jaccard: {np.mean(nn_ml_j):.3f}")

    print(f"\nAll outputs in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
