#!/usr/bin/env python3
"""
Diversity-aware DFT candidate nomination.

Replaces the pure-consensus top-25 selection with a 3-stage pipeline:
  1. Build a shortlist pool (top-N by ensemble RRF)
  2. Cluster the pool in PMTransformer embedding space
  3. Select 25 nominees via three diversity-aware strategies:
     A. Cluster-quota round-robin
     B. Maximal Marginal Relevance (MMR)
     C. Uncertainty-weighted cluster quota
  + a combined "best-of-3" final list.

Verification: SOAP pairwise distances and UMAP visualization.

Usage
-----
  python nominate_diverse_dft.py \
    --embeddings_path  /path/to/pmt_embeddings_qmof_unlabeled.npz \
    --prediction_csvs  exp364=/path/to/exp364/inference_predictions.csv \
                       exp370=/path/to/exp370/inference_predictions.csv \
                       exp371=/path/to/exp371/inference_predictions.csv \
                       smote_extra_trees=/path/to/extra_trees/test_predictions.csv \
                       smote_random_forest=/path/to/random_forest/test_predictions.csv \
    --output_dir       nomination_output \
    [--soap_embeddings_path /path/to/soap.npz] \
    [--old_nominees     /path/to/previous_nomination/FINAL_TOP25_diverse.txt] \
    [--pool_size 200] [--n_clusters 15] [--max_per_cluster 2] \
    [--mmr_lambdas 0.3 0.5 0.7] [--budget 25] [--rrf_k 60] [--seed 42]
"""

import os
import sys
import csv
import json
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Prediction loading — reuse logic from the ensemble-prediction pipeline
# ---------------------------------------------------------------------------


def infer_score_direction(mode_str):
    """regression / knn → lower = better;  multiclass → higher = better."""
    if not mode_str:
        return True
    mode_lower = str(mode_str).lower().strip()
    if any(tag in mode_lower for tag in ("regression", "knn", "sim_to_pos", "ensemble")):
        return True
    return False


def load_predictions_from_csv(csv_path, normalize_lower_is_better=True):
    """Load {cif_id: score} from test_predictions.csv / inference_predictions.csv.
    If *normalize_lower_is_better*, multiclass scores are inverted so that
    lower = better everywhere (consistent with NN regression).
    """
    preds = {}
    mode_str = "regression"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("cif_id", "").strip()
            if not cid:
                continue
            try:
                preds[cid] = float(row["score"])
                if "mode" in row and row["mode"]:
                    mode_str = row["mode"]
            except (KeyError, ValueError):
                continue
    if not preds:
        return preds
    if normalize_lower_is_better and not infer_score_direction(mode_str):
        max_s = max(preds.values())
        preds = {cid: max_s - s for cid, s in preds.items()}
    return preds


# ---------------------------------------------------------------------------
# Rank / fusion helpers
# ---------------------------------------------------------------------------


def score_to_rank(scores, lower_is_better=True):
    """Ordinal ranks (1 = best)."""
    arr = np.asarray(scores, dtype=float)
    order = np.argsort(arr) if lower_is_better else np.argsort(-arr)
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    return ranks


def _fill_missing(scores_arr, lower_is_better):
    arr = np.asarray(scores_arr, dtype=float)
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    fill = (np.nanmax(arr) + 1e9) if lower_is_better else (np.nanmin(arr) - 1e9)
    arr[mask] = fill
    return arr


def reciprocal_rank_fusion(models, cids, k=60):
    """RRF over *models* (each {cid: score}, lower = better).
    Returns {cid: score} where **lower = better**.
    """
    rrf = defaultdict(float)
    for _name, scores in models.items():
        arr = np.array([scores.get(c, np.nan) for c in cids])
        arr = _fill_missing(arr, lower_is_better=True)
        ranks = score_to_rank(arr, lower_is_better=True)
        for i, c in enumerate(cids):
            rrf[c] += 1.0 / (k + ranks[i])
    # Invert so lower = better
    mx = max(rrf.values()) if rrf else 0.0
    return {c: mx - rrf[c] for c in cids}


# ---------------------------------------------------------------------------
# Uncertainty / disagreement
# ---------------------------------------------------------------------------


def compute_uncertainty(models, cids, nn_names, ml_names):
    """For every CID compute rank-based uncertainty signals.

    Returns dict-of-dicts with keys:
      rank_std          – std of ranks across all models
      rank_range        – max rank – min rank
      nn_ml_disagreement – |mean NN rank – mean ML rank|

    All values are raw (not normalised).
    """
    # Pre-compute per-model rank arrays  (each array has len == len(cids))
    model_ranks = {}
    for name, scores in models.items():
        arr = np.array([scores.get(c, np.nan) for c in cids])
        arr = _fill_missing(arr, lower_is_better=True)
        model_ranks[name] = score_to_rank(arr, lower_is_better=True)

    rank_matrix = np.column_stack([model_ranks[n] for n in models])  # (N, M)
    std = rank_matrix.std(axis=1)
    rng = rank_matrix.max(axis=1) - rank_matrix.min(axis=1)

    nn_idx = [i for i, n in enumerate(models) if n in nn_names]
    ml_idx = [i for i, n in enumerate(models) if n in ml_names]
    if nn_idx and ml_idx:
        nn_mean = rank_matrix[:, nn_idx].mean(axis=1)
        ml_mean = rank_matrix[:, ml_idx].mean(axis=1)
        disagree = np.abs(nn_mean - ml_mean)
    else:
        disagree = np.zeros(len(cids))

    out = {}
    for i, c in enumerate(cids):
        out[c] = {
            "rank_std": float(std[i]),
            "rank_range": float(rng[i]),
            "nn_ml_disagreement": float(disagree[i]),
        }
    return out


def normalize_01(values):
    """Min-max normalise a 1-D array to [0, 1]."""
    arr = np.asarray(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_pool(embeddings, n_clusters, seed):
    """PCA-50 → KMeans.  Returns (labels, silhouette_score)."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    pca_dim = min(50, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=pca_dim, random_state=seed)
    reduced = pca.fit_transform(embeddings)

    n_clusters = min(n_clusters, embeddings.shape[0])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(reduced)

    sil = float("nan")
    if n_clusters > 1:
        from sklearn.metrics import silhouette_score as sil_fn
        sil = float(sil_fn(reduced, labels))
    return labels, sil


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


def strategy_cluster_quota(pool_cids, pool_quality, cluster_labels,
                           budget, max_per_cluster):
    """Round-robin: best quality per cluster, cycle through clusters
    sorted by best quality, cap at *max_per_cluster*."""
    # Group pool indices by cluster
    cluster_members = defaultdict(list)
    for i, c in enumerate(pool_cids):
        cluster_members[cluster_labels[i]].append(i)

    # Sort each cluster's members by quality (ascending = better)
    for cl in cluster_members:
        cluster_members[cl].sort(key=lambda i: pool_quality[i])

    # Sort clusters by best (lowest) quality score in each
    cluster_order = sorted(cluster_members.keys(),
                           key=lambda cl: pool_quality[cluster_members[cl][0]])

    selected = []
    selected_set = set()
    picked_per_cluster = defaultdict(int)
    pointer = {cl: 0 for cl in cluster_order}

    while len(selected) < budget:
        progress = False
        for cl in cluster_order:
            if len(selected) >= budget:
                break
            if picked_per_cluster[cl] >= max_per_cluster:
                continue
            members = cluster_members[cl]
            while pointer[cl] < len(members):
                idx = members[pointer[cl]]
                pointer[cl] += 1
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                    picked_per_cluster[cl] += 1
                    progress = True
                    break
            else:
                continue
        if not progress:
            # Relax cap
            max_per_cluster += 1

    return [pool_cids[i] for i in selected[:budget]]


def strategy_mmr(pool_cids, pool_quality_norm, dist_matrix, lam, budget):
    """Maximal Marginal Relevance.

    score_i = lambda * quality_i  +  (1-lambda) * min_dist(i, selected)
    quality_norm: 1 = best, 0 = worst  (inverted from rank).
    dist_matrix: pairwise cosine distance (higher = more diverse).
    """
    n = len(pool_cids)
    selected_idx = []
    remaining = set(range(n))

    # Seed with best quality
    first = int(np.argmax(pool_quality_norm))
    selected_idx.append(first)
    remaining.discard(first)

    for _ in range(budget - 1):
        if not remaining:
            break
        best_score = -np.inf
        best_idx = -1
        for i in remaining:
            min_d = min(dist_matrix[i, j] for j in selected_idx)
            score = lam * pool_quality_norm[i] + (1.0 - lam) * min_d
            if score > best_score:
                best_score = score
                best_idx = i
        selected_idx.append(best_idx)
        remaining.discard(best_idx)

    return [pool_cids[i] for i in selected_idx[:budget]]


def strategy_uncertainty_quota(pool_cids, pool_combined_score,
                               cluster_labels, budget, max_per_cluster):
    """Like cluster_quota but ranks within each cluster by a combined
    quality + uncertainty + disagreement score (lower = better)."""
    cluster_members = defaultdict(list)
    for i, c in enumerate(pool_cids):
        cluster_members[cluster_labels[i]].append(i)

    for cl in cluster_members:
        cluster_members[cl].sort(key=lambda i: pool_combined_score[i])

    cluster_order = sorted(cluster_members.keys(),
                           key=lambda cl: pool_combined_score[cluster_members[cl][0]])

    selected = []
    selected_set = set()
    picked_per_cluster = defaultdict(int)
    pointer = {cl: 0 for cl in cluster_order}

    while len(selected) < budget:
        progress = False
        for cl in cluster_order:
            if len(selected) >= budget:
                break
            if picked_per_cluster[cl] >= max_per_cluster:
                continue
            members = cluster_members[cl]
            while pointer[cl] < len(members):
                idx = members[pointer[cl]]
                pointer[cl] += 1
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                    picked_per_cluster[cl] += 1
                    progress = True
                    break
            else:
                continue
        if not progress:
            max_per_cluster += 1

    return [pool_cids[i] for i in selected[:budget]]


# ---------------------------------------------------------------------------
# SOAP diversity check
# ---------------------------------------------------------------------------


def soap_diversity(soap_emb, cids, selected_cids, label=""):
    """Compute pairwise cosine distances for selected set in SOAP space.
    Returns (min, mean, max) distance and the distance matrix.
    """
    from sklearn.metrics.pairwise import cosine_distances
    idx = [i for i, c in enumerate(cids) if c in set(selected_cids)]
    if len(idx) < 2:
        return None, None, None, None
    sub = soap_emb[idx]
    D = cosine_distances(sub)
    np.fill_diagonal(D, np.nan)
    vals = D[~np.isnan(D)]
    return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)), D


# ---------------------------------------------------------------------------
# UMAP visualisation
# ---------------------------------------------------------------------------


def plot_umap(all_emb, all_cids, pool_set, nominees_dict, old_nominees,
              output_dir, umap_cache_path=None, seed=42, emb_label="PMTransformer"):
    """2-D UMAP of the entire screening pool with overlays.

    nominees_dict: {strategy_name: [cid_list]}
    old_nominees: set of cids
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from umap import UMAP
    except ImportError:
        print("  [UMAP] matplotlib or umap-learn not installed — skipping plot.")
        return

    # Try cache
    coords = None
    if umap_cache_path and os.path.isfile(umap_cache_path):
        try:
            cache = np.load(umap_cache_path, allow_pickle=True)
            cached_ids = list(cache.get("cif_ids", cache.get("cids", [])))
            cached_coords = cache.get("coords", cache.get("coordinates", None))
            if cached_coords is not None and len(cached_ids) == len(all_cids):
                if set(cached_ids) == set(all_cids):
                    # Reorder to match all_cids
                    id2idx = {c: i for i, c in enumerate(cached_ids)}
                    reorder = [id2idx[c] for c in all_cids]
                    coords = cached_coords[reorder]
                    print("  [UMAP] Loaded from cache.")
        except Exception:
            pass

    if coords is None:
        print("  [UMAP] Fitting UMAP (may take a minute)...")
        reducer = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=seed)
        coords = reducer.fit_transform(all_emb)
        # Save cache
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        np.savez_compressed(os.path.join(plots_dir, "umap_cache.npz"),
                            cif_ids=np.array(all_cids), coords=coords)

    cid2idx = {c: i for i, c in enumerate(all_cids)}
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Plot 1: Pool + nominees ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=1, c="lightgrey", alpha=0.3, label="All candidates")

    pool_idx = [cid2idx[c] for c in pool_set if c in cid2idx]
    if pool_idx:
        ax.scatter(coords[pool_idx, 0], coords[pool_idx, 1],
                   s=8, c="steelblue", alpha=0.4, label="Shortlist pool")

    colours = ["red", "green", "purple", "orange", "cyan"]
    for ci, (sname, scids) in enumerate(nominees_dict.items()):
        idx = [cid2idx[c] for c in scids if c in cid2idx]
        if idx:
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       s=60, c=colours[ci % len(colours)], edgecolors="black",
                       linewidth=0.5, alpha=0.9, label=sname, zorder=5)

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Diversity-aware nominees — UMAP of {emb_label} embeddings")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "umap_diverse_nominees.png"), dpi=200)
    plt.close(fig)

    # --- Plot 2: Old vs new comparison ---
    combined_cids = set()
    for scids in nominees_dict.values():
        combined_cids.update(scids)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax_i, (title, cids_set, colour) in enumerate([
        ("Old nominees (consensus)", old_nominees, "red"),
        ("New nominees (diversity-aware)", combined_cids, "green"),
    ]):
        ax = axes[ax_i]
        ax.scatter(coords[:, 0], coords[:, 1], s=1, c="lightgrey", alpha=0.3)
        idx = [cid2idx[c] for c in cids_set if c in cid2idx]
        if idx:
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       s=80, c=colour, edgecolors="black", linewidth=0.7, zorder=5)
        ax.set_title(title, fontsize=13)
    fig.suptitle(f"Old vs New nominees on {emb_label} UMAP", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "umap_comparison_old_vs_new.png"), dpi=200)
    plt.close(fig)

    print(f"  [UMAP] Saved plots to {plots_dir}/")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_report(output_dir, args_dict, pool_size_actual, n_clusters, sil_score,
                 strategy_results, combined_cids, old_nominees,
                 soap_old, soap_new, pmtrans_old, pmtrans_new,
                 uncertainty_info, pool_cids, unc_data):
    """Write a Markdown diversity report."""
    path = os.path.join(output_dir, "diversity_report.md")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    overlap = set(combined_cids) & set(old_nominees)

    with open(path, "w", encoding="utf-8") as f:
        emb_label = args_dict.get('embedding_label', 'PMTransformer')
        f.write(f"# Diversity-Aware DFT Nomination Report ({emb_label})\n\n")
        f.write(f"**Generated**: {ts}\n")
        f.write(f"**Diversity space**: {emb_label} (key: `{args_dict.get('embedding_key', 'embeddings')}`)\n")
        f.write(f"**Budget**: {args_dict['budget']}\n")
        f.write(f"**Pool size**: {pool_size_actual} (requested {args_dict['pool_size']})\n")
        f.write(f"**Clusters**: {n_clusters} (silhouette = {sil_score:.3f})\n")
        f.write(f"**Seed**: {args_dict['seed']}\n\n")

        f.write("## Strategy Comparison\n\n")
        f.write("| Strategy | Description |\n|---|---|\n")
        for sname, scids in strategy_results.items():
            f.write(f"| {sname} | {len(scids)} nominees |\n")
        f.write(f"| **Combined** | {len(combined_cids)} nominees |\n\n")

        # Overlap with old
        f.write("## Old vs New Overlap\n\n")
        f.write(f"- Old nominees: {len(old_nominees)}\n")
        f.write(f"- New nominees (combined): {len(combined_cids)}\n")
        f.write(f"- Overlap: {len(overlap)}\n")
        if overlap:
            f.write(f"- Shared: {', '.join(sorted(overlap))}\n")
        f.write("\n")

        # Diversity metrics
        f.write("## Diversity Metrics (pairwise cosine distance)\n\n")
        f.write("| Set | Embedding | Min | Mean | Max |\n|---|---|---|---|---|\n")
        if pmtrans_old:
            f.write(f"| Old 25 | PMTransformer | {pmtrans_old[0]:.4f} | {pmtrans_old[1]:.4f} | {pmtrans_old[2]:.4f} |\n")
        if pmtrans_new:
            f.write(f"| New 25 | PMTransformer | {pmtrans_new[0]:.4f} | {pmtrans_new[1]:.4f} | {pmtrans_new[2]:.4f} |\n")
        if soap_old:
            f.write(f"| Old 25 | SOAP | {soap_old[0]:.4f} | {soap_old[1]:.4f} | {soap_old[2]:.4f} |\n")
        if soap_new:
            f.write(f"| New 25 | SOAP | {soap_new[0]:.4f} | {soap_new[1]:.4f} | {soap_new[2]:.4f} |\n")
        f.write("\n")

        # Per-strategy lists
        for sname, scids in strategy_results.items():
            f.write(f"## {sname}\n\n")
            f.write("| # | CIF ID |\n|---|---|\n")
            for j, cid in enumerate(scids, 1):
                f.write(f"| {j} | {cid} |\n")
            f.write("\n")

        # Combined final
        f.write("## Combined Final 25\n\n")
        f.write("| # | CIF ID | Strategies nominating | Avg RRF rank |\n|---|---|---|---|\n")
        # Build nomination counts
        nom_count = defaultdict(int)
        for sname, scids in strategy_results.items():
            for c in scids:
                nom_count[c] += 1
        for j, cid in enumerate(combined_cids, 1):
            unc = unc_data.get(cid, {})
            f.write(f"| {j} | {cid} | {nom_count.get(cid, 0)} | - |\n")
        f.write("\n")

        # Uncertainty profile of nominees
        f.write("## Uncertainty Profile of Combined Nominees\n\n")
        f.write("| CIF ID | Rank Std | Rank Range | NN-ML Disagreement |\n|---|---|---|---|\n")
        for cid in combined_cids:
            u = unc_data.get(cid, {})
            f.write(f"| {cid} | {u.get('rank_std', 0):.1f} | {u.get('rank_range', 0):.0f} | {u.get('nn_ml_disagreement', 0):.0f} |\n")
        f.write("\n")

    print(f"  Report saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Diversity-aware DFT candidate nomination (screening pool)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--embeddings_path", required=True,
                        help="Screening-pool embeddings .npz (keys: cif_ids, <embedding_key>)")
    parser.add_argument("--embedding_key", default="embeddings",
                        help="Key inside the .npz for the embedding matrix "
                             "(default: 'embeddings'; use 'soap_descriptors' for SOAP)")
    parser.add_argument("--embedding_label", default="PMTransformer",
                        help="Human-readable label for plots and report "
                             "(default: 'PMTransformer'; use 'SOAP' for SOAP runs)")
    parser.add_argument("--prediction_csvs", nargs="+", required=True,
                        help="model=path pairs, e.g. exp364=/path/to/inference_predictions.csv")
    parser.add_argument("--soap_embeddings_path", default=None,
                        help="Optional SOAP embeddings .npz (keys: cif_ids, embeddings)")
    parser.add_argument("--old_nominees", default=None,
                        help="Path to old FINAL_DFT_TOP25.txt for comparison")
    parser.add_argument("--umap_cache", default=None,
                        help="Path to existing umap_cache.npz (optional, speeds up plotting)")
    parser.add_argument("--output_dir", default="nomination_output",
                        help="Output directory (default: nomination_output)")
    parser.add_argument("--pool_size", type=int, default=200,
                        help="Shortlist pool size (default: 200)")
    parser.add_argument("--n_clusters", type=int, default=15,
                        help="Number of k-means clusters (default: 15)")
    parser.add_argument("--max_per_cluster", type=int, default=2,
                        help="Max picks per cluster in quota strategies (default: 2)")
    parser.add_argument("--mmr_lambdas", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="Lambda values for MMR (default: 0.3 0.5 0.7)")
    parser.add_argument("--budget", type=int, default=25,
                        help="Number of structures to nominate (default: 25)")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF smoothing parameter (default: 60)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    # Uncertainty weights for strategy C
    parser.add_argument("--alpha", type=float, default=0.50,
                        help="Weight for quality rank in strategy C (default: 0.50)")
    parser.add_argument("--beta", type=float, default=0.30,
                        help="Weight for uncertainty in strategy C (default: 0.30)")
    parser.add_argument("--gamma", type=float, default=0.20,
                        help="Weight for NN-ML disagreement in strategy C (default: 0.20)")
    # Model type annotations
    parser.add_argument("--nn_models", nargs="+",
                        default=["exp364", "exp370", "exp371"],
                        help="Which model names are NN (default: exp364 exp370 exp371)")
    parser.add_argument("--ml_models", nargs="+",
                        default=["smote_extra_trees", "smote_random_forest"],
                        help="Which model names are ML (default: smote_extra_trees smote_random_forest)")
    # Long-tail exploration
    parser.add_argument("--exploration_budget", type=int, default=5,
                        help="How many of the --budget slots to reserve for long-tail "
                             "exploration picks from OUTSIDE the pool (default: 5). "
                             "Set to 0 to disable.")
    parser.add_argument("--exploration_pool_lo", type=int, default=None,
                        help="Long-tail exploration zone start rank (default: pool_size+1)")
    parser.add_argument("--exploration_pool_hi", type=int, default=1000,
                        help="Long-tail exploration zone end rank (default: 1000)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. Load embeddings
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("  Diversity-aware DFT nomination")
    print("=" * 70)

    emb_data = np.load(args.embeddings_path, allow_pickle=True)
    all_cids = [str(c) for c in emb_data["cif_ids"]]  # force Python str (npz may store numpy.str_)
    emb_key = args.embedding_key
    if emb_key not in emb_data:
        avail = list(emb_data.keys())
        print(f"  ERROR: key '{emb_key}' not found in {args.embeddings_path}")
        print(f"  Available keys: {avail}")
        return 1
    all_emb = emb_data[emb_key]
    cid2emb_idx = {c: i for i, c in enumerate(all_cids)}
    print(f"  Loaded {len(all_cids)} structures, embedding dim = {all_emb.shape[1]}")
    print(f"  Embedding key: '{emb_key}'  label: {args.embedding_label}")
    print(f"  Example CIF IDs: {all_cids[:3]}")

    # SOAP (optional)
    soap_emb = None
    soap_cids = None
    if args.soap_embeddings_path and os.path.isfile(args.soap_embeddings_path):
        soap_data = np.load(args.soap_embeddings_path, allow_pickle=True)
        soap_cids = [str(c) for c in soap_data.get("cif_ids", soap_data.get("cids", []))]
        soap_emb = soap_data.get("embeddings",
                    soap_data.get("descriptors",
                    soap_data.get("soap_descriptors", None)))
        if soap_emb is not None:
            print(f"  Loaded SOAP embeddings: {soap_emb.shape}")
        else:
            print("  Warning: SOAP file found but no 'embeddings' or 'descriptors' key.")
    elif args.soap_embeddings_path:
        print(f"  Warning: SOAP file not found: {args.soap_embeddings_path}")

    # Old nominees
    old_nominees = set()
    if args.old_nominees and os.path.isfile(args.old_nominees):
        with open(args.old_nominees, "r", encoding="utf-8") as f:
            old_nominees = set(line.strip() for line in f if line.strip())
        print(f"  Loaded {len(old_nominees)} old nominees for comparison")

    # -------------------------------------------------------------------------
    # 2. Load predictions
    # -------------------------------------------------------------------------
    models = {}  # {name: {cid: score}}  lower = better
    for spec in args.prediction_csvs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name = os.path.splitext(os.path.basename(spec))[0]
            path = spec
        path = os.path.abspath(path)
        preds = load_predictions_from_csv(path, normalize_lower_is_better=True)
        if preds:
            models[name] = preds
            print(f"  Loaded {name}: {len(preds)} predictions from {os.path.basename(path)}")
        else:
            print(f"  Warning: no predictions loaded for {name} ({path})")

    if not models:
        print("ERROR: No model predictions loaded.")
        return 1

    nn_names = set(args.nn_models) & set(models.keys())
    ml_names = set(args.ml_models) & set(models.keys())
    print(f"  NN models: {sorted(nn_names)}  |  ML models: {sorted(ml_names)}")

    # -------------------------------------------------------------------------
    # DIAGNOSTIC: verify score directions & CID overlap
    # -------------------------------------------------------------------------
    print("\n  === DIAGNOSTICS (verify these!) ===")
    for mname, mscores in models.items():
        vals = list(mscores.values())
        n_overlap = len(set(mscores.keys()) & set(all_cids))
        sorted_cids_m = sorted(mscores.keys(), key=lambda c: mscores[c])
        print(f"  [{mname}] {len(vals)} predictions, "
              f"overlap with embeddings: {n_overlap}/{len(all_cids)}, "
              f"score range: [{min(vals):.4f}, {max(vals):.4f}], "
              f"median: {np.median(vals):.4f}")
        print(f"    Convention: lower = better (after normalisation)")
        print(f"    Top-5 (best):  {sorted_cids_m[:5]}")
        print(f"    Top-5 scores:  {[f'{mscores[c]:.4f}' for c in sorted_cids_m[:5]]}")
        print(f"    Bot-5 (worst): {sorted_cids_m[-5:]}")
        if n_overlap == 0:
            print(f"    *** WARNING: ZERO overlap between {mname} CIDs and embedding CIDs! ***")
            print(f"    CSV CID examples: {sorted_cids_m[:3]}")
            print(f"    Embedding CID examples: {all_cids[:3]}")
            print(f"    This likely means CID format mismatch — check your files!")
    print("  === END DIAGNOSTICS ===\n")

    # -------------------------------------------------------------------------
    # 3. Ensemble RRF → ranked list
    # -------------------------------------------------------------------------
    rrf_scores = reciprocal_rank_fusion(models, all_cids, k=args.rrf_k)
    rrf_sorted = sorted(all_cids, key=lambda c: rrf_scores[c])  # lower = better = first
    rrf_rank = {c: i + 1 for i, c in enumerate(rrf_sorted)}
    print(f"  RRF ensemble computed over {len(models)} models")

    # -------------------------------------------------------------------------
    # 4. Uncertainty signals
    # -------------------------------------------------------------------------
    unc_data = compute_uncertainty(models, all_cids, nn_names, ml_names)

    # -------------------------------------------------------------------------
    # 5. Build shortlist pool (top-N by RRF)
    # -------------------------------------------------------------------------
    pool_size = min(args.pool_size, len(all_cids))
    pool_cids = rrf_sorted[:pool_size]
    pool_set = set(pool_cids)
    pool_emb = np.array([all_emb[cid2emb_idx[c]] for c in pool_cids])
    pool_quality = np.array([rrf_scores[c] for c in pool_cids])  # lower = better
    print(f"  Shortlist pool: top {pool_size} by RRF")

    # Normalised quality for MMR (1 = best, 0 = worst)
    pool_quality_norm = 1.0 - normalize_01(pool_quality)

    # -------------------------------------------------------------------------
    # 6. Cluster the pool
    # -------------------------------------------------------------------------
    labels, sil_score = cluster_pool(pool_emb, args.n_clusters, args.seed)
    n_unique_clusters = len(set(labels))
    print(f"  Clustered into {n_unique_clusters} clusters (silhouette = {sil_score:.3f})")

    # -------------------------------------------------------------------------
    # 7. Pairwise cosine distance for MMR
    # -------------------------------------------------------------------------
    from sklearn.metrics.pairwise import cosine_distances
    dist_matrix = cosine_distances(pool_emb)
    # Normalise distances to [0,1] for MMR
    dmax = dist_matrix.max()
    if dmax > 0:
        dist_norm = dist_matrix / dmax
    else:
        dist_norm = dist_matrix

    # -------------------------------------------------------------------------
    # 8. Run selection strategies
    # -------------------------------------------------------------------------
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    strategy_results = {}

    # --- Strategy A: Cluster Quota ---
    print("\n  Strategy A: Cluster-quota round-robin")
    sel_a = strategy_cluster_quota(pool_cids, pool_quality, labels,
                                   args.budget, args.max_per_cluster)
    strategy_results["A_cluster_quota"] = sel_a
    cid2pool = {c: i for i, c in enumerate(pool_cids)}
    sel_a_clusters = set(labels[cid2pool[c]] for c in sel_a if c in cid2pool)
    print(f"    Selected {len(sel_a)} nominees across {len(sel_a_clusters)} clusters")

    # --- Strategy B: MMR (multiple lambdas) ---
    for lam in args.mmr_lambdas:
        sname = f"B_mmr_lambda{lam:.1f}"
        print(f"\n  Strategy B: MMR (lambda={lam})")
        sel_b = strategy_mmr(pool_cids, pool_quality_norm, dist_norm, lam, args.budget)
        strategy_results[sname] = sel_b
        print(f"    Selected {len(sel_b)} nominees")

    # --- Strategy C: Uncertainty-weighted cluster quota ---
    print(f"\n  Strategy C: Uncertainty-weighted cluster quota (alpha={args.alpha}, beta={args.beta}, gamma={args.gamma})")
    pool_unc_std = normalize_01([unc_data[c]["rank_std"] for c in pool_cids])
    pool_unc_dis = normalize_01([unc_data[c]["nn_ml_disagreement"] for c in pool_cids])
    # Combined: lower = better (quality is already lower=better, uncertainty we WANT high → invert)
    pool_combined = (
        args.alpha * normalize_01(pool_quality)                  # quality: lower rank = better
        + args.beta * (1.0 - pool_unc_std)                      # prefer HIGH uncertainty (inverted)
        + args.gamma * (1.0 - pool_unc_dis)                     # prefer HIGH disagreement (inverted)
    )
    sel_c = strategy_uncertainty_quota(pool_cids, pool_combined, labels,
                                       args.budget, args.max_per_cluster)
    strategy_results["C_uncertainty_quota"] = sel_c
    print(f"    Selected {len(sel_c)} nominees")

    # --- Strategy D: Long-tail exploration (hedge against pool being wrong) ---
    exploration_budget = min(args.exploration_budget, args.budget - 1)
    if exploration_budget > 0:
        explore_lo = args.exploration_pool_lo if args.exploration_pool_lo is not None else pool_size
        explore_hi = min(args.exploration_pool_hi, len(rrf_sorted))
        print(f"\n  Strategy D: Long-tail exploration ({exploration_budget} picks from ranks {explore_lo+1}–{explore_hi})")
        # Candidates: structures ranked explore_lo..explore_hi that are NOT in the pool
        tail_cids = [c for c in rrf_sorted[explore_lo:explore_hi] if c not in pool_set]
        if tail_cids:
            # Rank by NN-ML disagreement (highest first = most uncertain)
            tail_disagree = np.array([unc_data[c]["nn_ml_disagreement"] for c in tail_cids])
            tail_unc_std = np.array([unc_data[c]["rank_std"] for c in tail_cids])
            # Score: 60% disagreement + 40% rank_std (both higher = more interesting)
            tail_score = 0.6 * normalize_01(tail_disagree) + 0.4 * normalize_01(tail_unc_std)
            # Also enforce diversity: pick via mini-MMR within tail embeddings
            tail_emb = np.array([all_emb[cid2emb_idx[c]] for c in tail_cids])
            tail_dist = cosine_distances(tail_emb)
            td_max = tail_dist.max()
            if td_max > 0:
                tail_dist_norm = tail_dist / td_max
            else:
                tail_dist_norm = tail_dist

            # MMR with lambda=0.4 (heavy on diversity since we're hedging)
            sel_d = strategy_mmr(tail_cids, tail_score, tail_dist_norm,
                                 lam=0.4, budget=exploration_budget)
            strategy_results["D_longtail_exploration"] = sel_d
            print(f"    Selected {len(sel_d)} exploration picks from rank {explore_lo+1}–{explore_hi}")
            for j, cid in enumerate(sel_d, 1):
                print(f"      {j}. {cid}  (RRF rank={rrf_rank[cid]}, "
                      f"disagree={unc_data[cid]['nn_ml_disagreement']:.0f}, "
                      f"rank_std={unc_data[cid]['rank_std']:.0f})")
        else:
            print("    Warning: no tail candidates available — skipping.")
            exploration_budget = 0

    # -------------------------------------------------------------------------
    # 9. Combined "best-of-all" list
    # -------------------------------------------------------------------------
    print("\n  Combining strategies...")
    nom_count = defaultdict(int)
    nom_quality = {}
    # Count nominations from pool-based strategies (A, B, C) only
    pool_strategies = {k: v for k, v in strategy_results.items() if not k.startswith("D_")}
    for sname, scids in pool_strategies.items():
        for c in scids:
            nom_count[c] += 1
            if c not in nom_quality:
                nom_quality[c] = rrf_rank[c]

    # Reserve slots for long-tail exploration picks (guaranteed inclusion)
    exploration_picks = strategy_results.get("D_longtail_exploration", [])
    reserved = set(exploration_picks)
    main_budget = args.budget - len(exploration_picks)

    # Fill main slots from pool-based strategies (excluding exploration picks)
    pool_nominated = sorted(
        [c for c in nom_count if c not in reserved],
        key=lambda c: (-nom_count[c], nom_quality[c])
    )
    main_picks = pool_nominated[:main_budget]
    combined_cids = main_picks + list(exploration_picks)

    print(f"  Combined: {len(main_picks)} pool picks + {len(exploration_picks)} exploration picks = {len(combined_cids)}")
    if main_picks:
        print(f"  Pool pick nomination counts: max={max(nom_count[c] for c in main_picks)}, "
              f"min={min(nom_count[c] for c in main_picks)}")

    # -------------------------------------------------------------------------
    # 10. Save outputs
    # -------------------------------------------------------------------------
    print("\n  Saving outputs...")
    for sname, scids in strategy_results.items():
        fname = f"{sname}_top{args.budget}.txt"
        with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
            for c in scids:
                f.write(c + "\n")

    # Add COMBINED after writing per-strategy files (avoids double-counting in nom_count)
    strategy_results["COMBINED"] = combined_cids
    with open(os.path.join(output_dir, f"COMBINED_top{args.budget}.txt"), "w", encoding="utf-8") as f:
        for c in combined_cids:
            f.write(c + "\n")

    # Final CSV
    csv_path = os.path.join(output_dir, "FINAL_TOP25_diverse.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "cif_id", "rrf_rank", "strategies_nominating",
                         "rank_std", "rank_range", "nn_ml_disagreement", "cluster"])
        for j, cid in enumerate(combined_cids, 1):
            u = unc_data.get(cid, {})
            cl = labels[pool_cids.index(cid)] if cid in pool_set else -1
            writer.writerow([j, cid, rrf_rank.get(cid, -1), nom_count.get(cid, 0),
                             f"{u.get('rank_std', 0):.1f}",
                             f"{u.get('rank_range', 0):.0f}",
                             f"{u.get('nn_ml_disagreement', 0):.0f}",
                             cl])

    # Simple text list
    txt_path = os.path.join(output_dir, "FINAL_TOP25_diverse.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for c in combined_cids:
            f.write(c + "\n")

    # Shortlist pool CSV
    pool_csv = os.path.join(output_dir, "shortlist_pool.csv")
    with open(pool_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pool_rank", "cif_id", "rrf_rank", "cluster",
                         "rank_std", "nn_ml_disagreement"])
        for j, cid in enumerate(pool_cids, 1):
            u = unc_data.get(cid, {})
            writer.writerow([j, cid, rrf_rank.get(cid, -1), labels[j - 1],
                             f"{u.get('rank_std', 0):.1f}",
                             f"{u.get('nn_ml_disagreement', 0):.0f}"])

    print(f"  Saved {csv_path}")
    print(f"  Saved {txt_path}")

    # -------------------------------------------------------------------------
    # 11. Diversity verification
    # -------------------------------------------------------------------------
    print("\n  Computing diversity metrics...")

    # PMTransformer pairwise distances
    def pmtrans_diversity(cids_set):
        idx = [cid2emb_idx[c] for c in cids_set if c in cid2emb_idx]
        if len(idx) < 2:
            return None
        sub = all_emb[idx]
        D = cosine_distances(sub)
        np.fill_diagonal(D, np.nan)
        vals = D[~np.isnan(D)]
        return (float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)))

    pmtrans_old = pmtrans_diversity(old_nominees) if old_nominees else None
    pmtrans_new = pmtrans_diversity(combined_cids)

    if pmtrans_old:
        print(f"  PMTransformer distances — Old: min={pmtrans_old[0]:.4f} mean={pmtrans_old[1]:.4f} max={pmtrans_old[2]:.4f}")
    if pmtrans_new:
        print(f"  PMTransformer distances — New: min={pmtrans_new[0]:.4f} mean={pmtrans_new[1]:.4f} max={pmtrans_new[2]:.4f}")

    # SOAP diversity
    soap_old_stats = None
    soap_new_stats = None
    if soap_emb is not None and soap_cids is not None:
        mn, me, mx, _ = soap_diversity(soap_emb, soap_cids, list(old_nominees))
        soap_old_stats = (mn, me, mx) if mn is not None else None
        mn, me, mx, _ = soap_diversity(soap_emb, soap_cids, combined_cids)
        soap_new_stats = (mn, me, mx) if mn is not None else None
        if soap_old_stats:
            print(f"  SOAP distances — Old: min={soap_old_stats[0]:.4f} mean={soap_old_stats[1]:.4f} max={soap_old_stats[2]:.4f}")
        if soap_new_stats:
            print(f"  SOAP distances — New: min={soap_new_stats[0]:.4f} mean={soap_new_stats[1]:.4f} max={soap_new_stats[2]:.4f}")

    # -------------------------------------------------------------------------
    # 12. UMAP plots
    # -------------------------------------------------------------------------
    plot_umap(all_emb, all_cids, pool_set,
              {k: v for k, v in strategy_results.items() if k == "COMBINED"},
              old_nominees, output_dir,
              umap_cache_path=args.umap_cache, seed=args.seed,
              emb_label=args.embedding_label)

    # -------------------------------------------------------------------------
    # 13. Write full report
    # -------------------------------------------------------------------------
    write_report(output_dir, vars(args), pool_size, n_unique_clusters, sil_score,
                 strategy_results, combined_cids, old_nominees,
                 soap_old_stats, soap_new_stats, pmtrans_old, pmtrans_new,
                 {}, pool_cids, unc_data)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DONE — Diversity-aware nomination complete")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  Combined top-{args.budget}: {txt_path}")
    overlap = set(combined_cids) & set(old_nominees)
    if old_nominees:
        print(f"  Overlap with old nominees: {len(overlap)}/{len(old_nominees)}")
    if pmtrans_new and pmtrans_old:
        delta = pmtrans_new[1] - pmtrans_old[1]
        print(f"  PMTransformer mean distance: {pmtrans_old[1]:.4f} (old) → {pmtrans_new[1]:.4f} (new)  "
              f"({'↑' if delta > 0 else '↓'} {abs(delta):.4f})")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
