#!/usr/bin/env python3
"""
SOAP Structural Validation — Independent Confirmation of Data Split
================================================================

Uses SOAP (Smooth Overlap of Atomic Positions) descriptors — computed purely
from crystal geometry — to provide NN-independent structural evidence that
the data split is meaningful.  This directly addresses the circularity concern:
the split was designed in NN embedding space, so we validate it with a
completely independent structural representation.

Pipeline
--------
  Stage 1  Compute per-MOF average SOAP descriptors from CIF files
           → cached in  <output_dir>/soap_descriptors.npz

  Stage 2  Quantitative analyses  (always runs; fast once SOAP is cached)
           A. Split coverage validation  — are test positives structurally
              near train positives even in SOAP space?
           B. Structure–bandgap correlation — does SOAP similarity predict
              bandgap similarity?  (validates the fundamental assumption)
           C. Applicability domain  — are ensemble top-K predictions
              structurally grounded, or extrapolations?
           D. Representation agreement (Mantel test) — one-number summary
              of how much NN embeddings capture structural information

Usage  (cluster)
----------------
  python figures/soap_validation.py \\
      --cif_dir data/raw/cif \\
      --merged_embeddings figures_output/pretrained_embeddings/all_embeddings.npz \\
      --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --output_dir figures_output/soap_validation

  # With ensemble predictions for applicability domain:
  python figures/soap_validation.py \\
      --cif_dir data/raw/cif --merged_embeddings figures_output/pretrained_embeddings/all_embeddings.npz \\
      --labeled_splits_dir data/splits/strategy_d_farthest_point \\
      --nominations data/unlabeled/nomination-SOAP/FINAL_TOP25_diverse.txt \\
      --output_dir figures_output/soap_validation

Requirements
------------
  pip install dscribe ase numpy matplotlib scipy
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from scipy import stats as spstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────
SOAP_RCUT = 6.0       # Angstrom
SOAP_NMAX = 4
SOAP_LMAX = 4
SOAP_SIGMA = 0.5
SOAP_PERIODIC = True

BANDGAP_THRESHOLD = 1.0  # eV — positive class boundary

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
        "legend.fontsize":   7,
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
    print(f"      Saved {name}.png / .svg / .pdf")


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1 — Compute SOAP descriptors  (unchanged, with caching)
# ══════════════════════════════════════════════════════════════════════
def discover_cif_files(cif_dir):
    """Return dict {cif_id: full_path} for all .cif files in cif_dir."""
    cifs = {}
    for fn in os.listdir(cif_dir):
        if fn.endswith(".cif"):
            cid = fn[:-4]
            cifs[cid] = os.path.join(cif_dir, fn)
    return cifs


def compute_soap_descriptors(cif_dir, cif_ids, output_path,
                             rcut=SOAP_RCUT, nmax=SOAP_NMAX,
                             lmax=SOAP_LMAX, sigma=SOAP_SIGMA):
    """Compute average SOAP descriptor for each MOF from its CIF file.

    Uses dscribe SOAP with average='inner' → one fixed-length vector per
    structure regardless of atom count.

    Returns (ordered_cif_ids, soap_matrix [N, D]).
    """
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
        species=species,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        sigma=sigma,
        periodic=SOAP_PERIODIC,
        average="inner",
        sparse=False,
    )
    soap_dim = soap.get_number_of_features()
    print(f"    SOAP descriptor dimension: {soap_dim}")

    print("    Computing SOAP descriptors ...")
    ordered_ids = []
    soap_list = []
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
            elapsed = now - t0
            rate = done / elapsed if elapsed > 0 else 0
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
        params=np.array([rcut, nmax, lmax, sigma]),
    )
    print(f"    SOAP cache saved -> {output_path}")
    return ordered_ids, soap_matrix


def load_soap_cache(cache_path):
    """Load cached SOAP descriptors."""
    data = np.load(cache_path, allow_pickle=True)
    cif_ids = list(data["cif_ids"])
    soap_matrix = data["soap_descriptors"]
    params = data["params"] if "params" in data else None
    print(f"    Loaded SOAP cache: {len(cif_ids)} MOFs, dim={soap_matrix.shape[1]}")
    if params is not None:
        print(f"    Params: r_cut={params[0]}, n_max={int(params[1])}, "
              f"l_max={int(params[2])}, sigma={params[3]}")
    return cif_ids, soap_matrix


# ══════════════════════════════════════════════════════════════════════
#  Metadata loading
# ══════════════════════════════════════════════════════════════════════
def load_npz_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "cif_ids":    list(data["cif_ids"]),
        "embeddings": data["embeddings"],
        "bandgaps":   data["bandgaps"] if "bandgaps" in data else None,
        "splits":     list(data["splits"]) if "splits" in data else None,
        "is_labeled": data["is_labeled"] if "is_labeled" in data else None,
    }


def load_split_labels(splits_dir):
    """Load bandgap labels and split assignments from JSON files."""
    labels, assignments = {}, {}
    for split_name in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{split_name}_bandgaps_regression.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        for cid, bg in d.items():
            labels[cid] = float(bg)
            assignments[cid] = split_name
        print(f"      {split_name}: {len(d)} MOFs")
    return labels, assignments


def load_nominations(filepath):
    """Load a top-K prediction file (one CIF ID per line)."""
    cids = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                cids.append(line)
    print(f"    Loaded {len(cids)} ensemble predictions from {filepath}")
    return cids


# ══════════════════════════════════════════════════════════════════════
#  Similarity computation
# ══════════════════════════════════════════════════════════════════════
def cosine_similarity_matrix(A, B=None):
    """Compute cosine similarity between rows of A and B.

    Returns [len(A), len(B)] matrix with values in [-1, 1].
    If B is None, computes A vs A.
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    if B is None:
        B_norm = A_norm
    else:
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def soap_similarity_matrix(A, B=None):
    """Normalized SOAP kernel: inner product of L2-normalized SOAP vectors.

    SOAP descriptors are non-negative (power spectrum), so the normalized
    kernel is in [0, 1].  This is the standard SOAP similarity metric.
    """
    return cosine_similarity_matrix(A, B)


# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS A — Split Coverage Validation
# ══════════════════════════════════════════════════════════════════════
def analysis_a_split_coverage(soap_cids, soap_matrix, nn_cids, nn_embs,
                              labeled_bg, labeled_splits, threshold,
                              output_dir):
    """For each test positive, compare SOAP vs NN similarity to nearest
    train positive.  If both agree on coverage → the split is structurally
    meaningful, not an NN artifact."""

    print("\n  ── Analysis A: Split Coverage Validation ──")

    # Identify positives by split
    train_pos_cids = [c for c, s in labeled_splits.items()
                      if s == "train" and labeled_bg[c] < threshold]
    val_pos_cids   = [c for c, s in labeled_splits.items()
                      if s == "val" and labeled_bg[c] < threshold]
    test_pos_cids  = [c for c, s in labeled_splits.items()
                      if s == "test" and labeled_bg[c] < threshold]

    print(f"    Train positives: {len(train_pos_cids)}")
    print(f"    Val positives:   {len(val_pos_cids)}")
    print(f"    Test positives:  {len(test_pos_cids)}")

    if not test_pos_cids or not train_pos_cids:
        print("    WARNING: No test or train positives found — skipping")
        return None

    # Build index lookups
    soap_idx = {c: i for i, c in enumerate(soap_cids)}
    nn_idx   = {c: i for i, c in enumerate(nn_cids)}

    # Filter to CIDs present in both representations
    eval_cids = test_pos_cids + val_pos_cids  # evaluate coverage for both
    eval_cids = [c for c in eval_cids if c in soap_idx and c in nn_idx]
    anchor_cids = [c for c in train_pos_cids if c in soap_idx and c in nn_idx]

    if not eval_cids or not anchor_cids:
        print("    WARNING: Missing CIDs in SOAP/NN — skipping")
        return None

    # Extract matrices
    eval_soap  = soap_matrix[[soap_idx[c] for c in eval_cids]]
    anchor_soap = soap_matrix[[soap_idx[c] for c in anchor_cids]]
    eval_nn   = nn_embs[[nn_idx[c] for c in eval_cids]]
    anchor_nn = nn_embs[[nn_idx[c] for c in anchor_cids]]

    # Compute similarity matrices
    soap_sim = soap_similarity_matrix(eval_soap, anchor_soap)
    nn_sim   = cosine_similarity_matrix(eval_nn, anchor_nn)

    # Per-eval-positive: nearest train positive in each space
    results = []
    for i, cid in enumerate(eval_cids):
        split = labeled_splits[cid]
        bg = labeled_bg[cid]
        # SOAP
        s_best_j = int(np.argmax(soap_sim[i]))
        s_best_sim = float(soap_sim[i, s_best_j])
        s_best_cid = anchor_cids[s_best_j]
        # NN
        n_best_j = int(np.argmax(nn_sim[i]))
        n_best_sim = float(nn_sim[i, n_best_j])
        n_best_cid = anchor_cids[n_best_j]

        results.append({
            "cif_id": cid,
            "split": split,
            "bandgap": round(bg, 4),
            "soap_nearest_train_pos": s_best_cid,
            "soap_similarity": round(s_best_sim, 4),
            "nn_nearest_train_pos": n_best_cid,
            "nn_cosine_similarity": round(n_best_sim, 4),
            "same_nearest": s_best_cid == n_best_cid,
        })

    # Print table
    print(f"\n    {'CIF ID':<35} {'Split':>5}  {'BG':>5}  "
          f"{'SOAP sim':>8}  {'NN sim':>8}  {'SOAP anchor':<30}  {'NN anchor':<30}  {'Same?':>5}")
    print("    " + "─" * 140)
    for r in results:
        print(f"    {r['cif_id']:<35} {r['split']:>5}  {r['bandgap']:>5.3f}  "
              f"{r['soap_similarity']:>8.4f}  {r['nn_cosine_similarity']:>8.4f}  "
              f"{r['soap_nearest_train_pos']:<30}  {r['nn_nearest_train_pos']:<30}  "
              f"{'Y' if r['same_nearest'] else 'N':>5}")

    # Correlation between SOAP and NN similarities
    soap_sims = [r["soap_similarity"] for r in results]
    nn_sims   = [r["nn_cosine_similarity"] for r in results]
    if len(results) >= 3:
        rho, p = spstats.spearmanr(soap_sims, nn_sims)
        print(f"\n    Spearman ρ (SOAP sim vs NN sim): {rho:.3f}  (p={p:.4f})")
    else:
        rho, p = np.nan, np.nan

    # Summary stats
    soap_mean = np.mean(soap_sims)
    nn_mean   = np.mean(nn_sims)
    n_same    = sum(1 for r in results if r["same_nearest"])
    print(f"    Mean SOAP similarity to nearest train positive: {soap_mean:.4f}")
    print(f"    Mean NN   similarity to nearest train positive: {nn_mean:.4f}")
    print(f"    Same nearest anchor in both spaces: {n_same}/{len(results)}")

    # ── Figure: Grouped bar chart ─────────────────────────────────────
    test_results = [r for r in results if r["split"] == "test"]
    val_results  = [r for r in results if r["split"] == "val"]

    if test_results:
        _plot_coverage_bars(test_results + val_results, output_dir)

    # ── Figure: Heatmap of test-pos × train-pos SOAP similarity ──────
    test_eval_mask = [i for i, c in enumerate(eval_cids)
                      if labeled_splits[c] == "test"]
    if len(test_eval_mask) >= 2:
        _plot_coverage_heatmap(
            soap_sim[test_eval_mask, :],
            nn_sim[test_eval_mask, :],
            [eval_cids[i] for i in test_eval_mask],
            anchor_cids,
            output_dir)

    # Save results JSON
    out = {
        "per_positive": results,
        "summary": {
            "mean_soap_sim": round(soap_mean, 4),
            "mean_nn_sim": round(nn_mean, 4),
            "spearman_rho": round(rho, 4) if not np.isnan(rho) else None,
            "spearman_p":   round(p, 4)   if not np.isnan(p) else None,
            "same_nearest_count": n_same,
            "total_evaluated": len(results),
        },
    }
    jp = os.path.join(output_dir, "analysis_a_coverage.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"    Results -> {jp}")

    return out


def _plot_coverage_bars(results, output_dir):
    """Grouped bar chart: per test/val positive SOAP vs NN similarity."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    n = len(results)
    x = np.arange(n)
    w = 0.35

    soap_vals = [r["soap_similarity"] for r in results]
    nn_vals   = [r["nn_cosine_similarity"] for r in results]
    labels    = [f"{r['cif_id'][:18]}..." if len(r['cif_id']) > 18
                 else r['cif_id'] for r in results]

    ax.bar(x - w/2, soap_vals, w, label="SOAP similarity",
           color="#4292c6", edgecolor="white", linewidth=0.4)
    ax.bar(x + w/2, nn_vals, w, label="NN cosine similarity",
           color="#fd8d3c", edgecolor="white", linewidth=0.4)

    # Reference line at 0.55 (coverage threshold)
    ax.axhline(0.55, color="#e41a1c", ls="--", lw=1.0, alpha=0.7,
               label="Coverage threshold (0.55)")

    ax.set_ylabel("Similarity to nearest\ntrain positive")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.7", fontsize=7)

    # Annotate split membership with color
    for i, r in enumerate(results):
        color = "#4292c6" if r["split"] == "test" else "#fd8d3c"
        ax.annotate(r["split"][0].upper(), (i, -0.06),
                    ha="center", va="top", fontsize=5.5, fontweight="bold",
                    color=color, annotation_clip=False)

    ax.set_title("Split Coverage: SOAP vs. NN Similarity to Nearest "
                 "Training Positive", fontweight="bold", fontsize=9, pad=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    _save_panel(fig, output_dir, "analysis_a_coverage_bars")
    plt.close()


def _plot_coverage_heatmap(soap_sim_sub, nn_sim_sub, test_cids, train_cids,
                           output_dir):
    """Side-by-side heatmaps: test-pos × train-pos similarity in SOAP vs NN."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    short_test  = [c[:18] for c in test_cids]
    short_train = [c[:14] for c in train_cids]

    for ax, mat, title in [(ax1, soap_sim_sub, "SOAP similarity"),
                           (ax2, nn_sim_sub, "NN cosine similarity")]:
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_yticks(range(len(short_test)))
        ax.set_yticklabels(short_test, fontsize=6)
        ax.set_ylabel("Test positive", fontsize=7)

        # Only show a subset of x-ticks if too many train positives
        n_train = len(short_train)
        if n_train > 20:
            step = max(1, n_train // 15)
            xt = list(range(0, n_train, step))
            ax.set_xticks(xt)
            ax.set_xticklabels([short_train[i] for i in xt],
                               rotation=90, fontsize=5)
        else:
            ax.set_xticks(range(n_train))
            ax.set_xticklabels(short_train, rotation=90, fontsize=5.5)

        ax.set_xlabel("Training positive", fontsize=7)
        ax.set_title(title, fontsize=8, fontweight="bold")

    cbar = plt.colorbar(im, ax=[ax1, ax2], shrink=0.8, aspect=20, pad=0.03)
    cbar.set_label("Similarity", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.4)

    fig.suptitle("Test Positive Coverage: SOAP vs. NN",
                 fontweight="bold", fontsize=9, y=1.03)
    plt.tight_layout()
    _save_panel(fig, output_dir, "analysis_a_coverage_heatmap")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS B — Structure–Bandgap Correlation
# ══════════════════════════════════════════════════════════════════════
def analysis_b_structure_bandgap(soap_cids, soap_matrix, labeled_bg,
                                 threshold, output_dir,
                                 n_sample_negatives=500, seed=42):
    """Test whether structurally similar MOFs (by SOAP) have similar
    bandgaps.  This validates the fundamental assumption that structure
    determines bandgap, justifying structure-based prediction.

    Computes all pairwise SOAP similarities among:
      - All positives (bg < threshold): small set → all pairs
      - A random sample of negatives for comparison context
    """

    print("\n  ── Analysis B: Structure–Bandgap Correlation ──")

    soap_idx = {c: i for i, c in enumerate(soap_cids)}
    rng = np.random.RandomState(seed)

    # Positives with SOAP
    pos_cids = [c for c in soap_cids if c in labeled_bg
                and labeled_bg[c] < threshold]
    # All labeled with SOAP
    labeled_cids = [c for c in soap_cids if c in labeled_bg]

    print(f"    Positives in SOAP set: {len(pos_cids)}")
    print(f"    All labeled in SOAP set: {len(labeled_cids)}")

    if len(pos_cids) < 3:
        print("    WARNING: Too few positives for correlation — skipping")
        return None

    # ── Part 1: positive–positive pairs ───────────────────────────────
    pos_soap = soap_matrix[[soap_idx[c] for c in pos_cids]]
    pos_bgs  = np.array([labeled_bg[c] for c in pos_cids])

    sim_mat = soap_similarity_matrix(pos_soap)
    n_pos = len(pos_cids)

    # Upper triangle pairs (exclude diagonal)
    triu_i, triu_j = np.triu_indices(n_pos, k=1)
    pp_sims   = sim_mat[triu_i, triu_j]
    pp_delta  = np.abs(pos_bgs[triu_i] - pos_bgs[triu_j])
    n_pp = len(pp_sims)

    rho_pp, p_pp = spstats.spearmanr(pp_sims, pp_delta)
    print(f"    Positive–positive pairs: {n_pp}")
    print(f"    Spearman ρ(SOAP_sim, |ΔBG|): {rho_pp:.4f}  (p={p_pp:.2e})")

    # ── Part 2: broader labeled sample ────────────────────────────────
    neg_cids = [c for c in labeled_cids if labeled_bg[c] >= threshold]
    n_neg_sample = min(n_sample_negatives, len(neg_cids))
    if n_neg_sample > 0:
        sampled_neg = list(rng.choice(neg_cids, n_neg_sample, replace=False))
    else:
        sampled_neg = []
    broad_cids = pos_cids + sampled_neg
    broad_soap = soap_matrix[[soap_idx[c] for c in broad_cids]]
    broad_bgs  = np.array([labeled_bg[c] for c in broad_cids])

    sim_broad = soap_similarity_matrix(broad_soap)
    bi, bj = np.triu_indices(len(broad_cids), k=1)
    broad_sims  = sim_broad[bi, bj]
    broad_delta = np.abs(broad_bgs[bi] - broad_bgs[bj])

    rho_broad, p_broad = spstats.spearmanr(broad_sims, broad_delta)
    print(f"    Broad sample pairs ({len(broad_cids)} MOFs): {len(broad_sims)}")
    print(f"    Spearman ρ(SOAP_sim, |ΔBG|): {rho_broad:.4f}  (p={p_broad:.2e})")

    # ── Figure: Hexbin + binned means ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.5))

    # Panel 1: positive-positive
    ax1.hexbin(pp_sims, pp_delta, gridsize=30, cmap="Blues",
               mincnt=1, linewidths=0.3)
    _add_binned_mean(ax1, pp_sims, pp_delta, n_bins=8, color="#e41a1c")
    ax1.set_xlabel("SOAP similarity")
    ax1.set_ylabel("|Δ Bandgap| (eV)")
    ax1.set_title(f"(a) Positive–positive pairs (n={n_pp})\n"
                  f"Spearman ρ = {rho_pp:.3f}",
                  fontweight="bold", fontsize=8)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # Panel 2: broad (positives + sampled negatives)
    hb = ax2.hexbin(broad_sims, broad_delta, gridsize=40, cmap="Blues",
                    mincnt=1, linewidths=0.3)
    _add_binned_mean(ax2, broad_sims, broad_delta, n_bins=10,
                     color="#e41a1c")
    ax2.set_xlabel("SOAP similarity")
    ax2.set_ylabel("|Δ Bandgap| (eV)")
    ax2.set_title(f"(b) All labeled sample (n={len(broad_sims):,})\n"
                  f"Spearman ρ = {rho_broad:.3f}",
                  fontweight="bold", fontsize=8)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    cbar = plt.colorbar(hb, ax=ax2, shrink=0.85, aspect=20, pad=0.03)
    cbar.set_label("Count", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.4)

    fig.suptitle("Structure–Bandgap Relationship in SOAP Space",
                 fontweight="bold", fontsize=9, y=1.02)
    plt.tight_layout()
    _save_panel(fig, output_dir, "analysis_b_structure_bandgap")
    plt.close()

    # Save
    out = {
        "positive_pairs": {
            "n_pairs": n_pp,
            "spearman_rho": round(rho_pp, 4),
            "spearman_p": float(f"{p_pp:.2e}"),
        },
        "broad_sample": {
            "n_mofs": len(broad_cids),
            "n_positives": len(pos_cids),
            "n_negatives_sampled": n_neg_sample,
            "n_pairs": len(broad_sims),
            "spearman_rho": round(rho_broad, 4),
            "spearman_p": float(f"{p_broad:.2e}"),
        },
    }
    jp = os.path.join(output_dir, "analysis_b_structure_bandgap.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"    Results -> {jp}")
    return out


def _add_binned_mean(ax, x, y, n_bins=8, color="red"):
    """Add binned mean line with error bars to a scatter/hexbin plot."""
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    means, stds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= 3:
            means.append(y[mask].mean())
            stds.append(y[mask].std() / np.sqrt(mask.sum()))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    means, stds = np.array(means), np.array(stds)
    valid = np.isfinite(means)
    ax.errorbar(centers[valid], means[valid], yerr=stds[valid],
                color=color, lw=1.5, capsize=3, capthick=0.8,
                marker="o", markersize=4, zorder=10,
                label="Binned mean ± SE")
    ax.legend(loc="upper right", fontsize=7)


# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS C — Applicability Domain for Ensemble Predictions
# ══════════════════════════════════════════════════════════════════════
def analysis_c_applicability_domain(soap_cids, soap_matrix,
                                    nn_cids, nn_embs,
                                    labeled_bg, labeled_splits,
                                    nomination_cids, threshold, output_dir):
    """For each ensemble top-K prediction, evaluate structural novelty
    relative to training data using SOAP.

    Reports whether each discovery candidate is:
      - structurally similar to a known training positive  (high confidence)
      - structurally similar to training data but no positives (moderate)
      - structurally novel — far from all training data (low confidence)
    """

    print("\n  ── Analysis C: Applicability Domain ──")

    soap_idx = {c: i for i, c in enumerate(soap_cids)}
    nn_idx   = {c: i for i, c in enumerate(nn_cids)}

    # Training MOFs
    train_cids = [c for c, s in labeled_splits.items() if s == "train"]
    train_pos_cids = [c for c in train_cids if labeled_bg[c] < threshold]

    # Filter to those present in SOAP
    train_soap_cids     = [c for c in train_cids if c in soap_idx]
    train_pos_soap_cids = [c for c in train_pos_cids if c in soap_idx]
    nomination_soap_cids = [c for c in nomination_cids if c in soap_idx]

    print(f"    Train MOFs in SOAP: {len(train_soap_cids)}  "
          f"(positives: {len(train_pos_soap_cids)})")
    print(f"    Ensemble candidates in SOAP: {len(nomination_soap_cids)} / "
          f"{len(nomination_cids)}")

    if not nomination_soap_cids or not train_soap_cids:
        print("    WARNING: Insufficient data — skipping")
        return None

    nomination_soap = soap_matrix[[soap_idx[c] for c in nomination_soap_cids]]
    train_soap  = soap_matrix[[soap_idx[c] for c in train_soap_cids]]

    # Similarity to all training data
    sim_to_train = soap_similarity_matrix(nomination_soap, train_soap)
    max_sim_any  = sim_to_train.max(axis=1)
    nn_any_cids  = [train_soap_cids[int(j)] for j in sim_to_train.argmax(axis=1)]

    # Similarity to training positives
    if train_pos_soap_cids:
        train_pos_soap = soap_matrix[[soap_idx[c] for c in train_pos_soap_cids]]
        sim_to_pos = soap_similarity_matrix(nomination_soap, train_pos_soap)
        max_sim_pos = sim_to_pos.max(axis=1)
        nn_pos_cids = [train_pos_soap_cids[int(j)]
                       for j in sim_to_pos.argmax(axis=1)]
    else:
        max_sim_pos = np.zeros(len(nomination_soap_cids))
        nn_pos_cids = ["N/A"] * len(nomination_soap_cids)

    # Also compute NN-based similarity for comparison
    nomination_nn_cids = [c for c in nomination_soap_cids if c in nn_idx]
    nn_sim_any = {}
    if nomination_nn_cids:
        train_nn_cids = [c for c in train_cids if c in nn_idx]
        if train_nn_cids:
            nom_nn = nn_embs[[nn_idx[c] for c in nomination_nn_cids]]
            tr_nn  = nn_embs[[nn_idx[c] for c in train_nn_cids]]
            s_nn   = cosine_similarity_matrix(nom_nn, tr_nn)
            for i, c in enumerate(nomination_nn_cids):
                nn_sim_any[c] = float(s_nn[i].max())

    # Build results table
    results = []
    for i, cid in enumerate(nomination_soap_cids):
        s_any = float(max_sim_any[i])
        s_pos = float(max_sim_pos[i])
        s_nn  = nn_sim_any.get(cid, np.nan)

        # Classification
        if s_pos >= 0.5:
            conf = "high"
        elif s_any >= 0.5:
            conf = "moderate"
        else:
            conf = "low"

        results.append({
            "rank": nomination_cids.index(cid) + 1 if cid in nomination_cids else -1,
            "cif_id": cid,
            "soap_sim_nearest_train": round(s_any, 4),
            "soap_nearest_train_cid": nn_any_cids[i],
            "soap_sim_nearest_train_pos": round(s_pos, 4),
            "soap_nearest_train_pos_cid": nn_pos_cids[i],
            "nn_sim_nearest_train": round(s_nn, 4) if not np.isnan(s_nn) else None,
            "confidence": conf,
        })

    results.sort(key=lambda r: r["rank"])

    # Print summary
    print(f"\n    {'Rank':>4}  {'CIF ID':<35}  {'SOAP→train':>10}  "
          f"{'SOAP→pos':>9}  {'NN→train':>9}  {'Confidence':>10}")
    print("    " + "─" * 110)
    for r in results:
        nn_str = f"{r['nn_sim_nearest_train']:.4f}" if r["nn_sim_nearest_train"] else "  N/A"
        print(f"    {r['rank']:>4}  {r['cif_id']:<35}  "
              f"{r['soap_sim_nearest_train']:>10.4f}  "
              f"{r['soap_sim_nearest_train_pos']:>9.4f}  "
              f"{nn_str:>9}  {r['confidence']:>10}")

    n_high = sum(1 for r in results if r["confidence"] == "high")
    n_mod  = sum(1 for r in results if r["confidence"] == "moderate")
    n_low  = sum(1 for r in results if r["confidence"] == "low")
    print(f"\n    Confidence: high={n_high}, moderate={n_mod}, low={n_low}")

    # ── Figure: bar chart of SOAP similarities for top-K ──────────────
    if results:
        _plot_applicability_bars(results, output_dir)

    out = {
        "per_prediction": results,
        "summary": {
            "n_predictions": len(results),
            "n_high_confidence": n_high,
            "n_moderate_confidence": n_mod,
            "n_low_confidence": n_low,
            "mean_soap_sim_to_train": round(float(max_sim_any.mean()), 4),
            "mean_soap_sim_to_train_pos": round(float(max_sim_pos.mean()), 4),
        },
    }
    jp = os.path.join(output_dir, "analysis_c_applicability.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"    Results -> {jp}")
    return out


def _plot_applicability_bars(results, output_dir):
    """Horizontal bar chart: SOAP similarity to nearest train/train-pos."""
    fig, ax = plt.subplots(figsize=(5.0, max(2.5, 0.3 * len(results))))

    y = np.arange(len(results))
    labels = [f"#{r['rank']}  {r['cif_id'][:22]}" for r in results]
    s_any = [r["soap_sim_nearest_train"] for r in results]
    s_pos = [r["soap_sim_nearest_train_pos"] for r in results]
    confs = [r["confidence"] for r in results]

    conf_colors = {"high": "#2ca02c", "moderate": "#ff7f0e", "low": "#d62728"}
    bar_colors = [conf_colors[c] for c in confs]

    ax.barh(y, s_any, height=0.7, color="#a6bddb", edgecolor="white",
            linewidth=0.3, label="→ nearest train (any)", zorder=1)
    ax.scatter(s_pos, y, c=bar_colors, s=30, zorder=3, edgecolors="black",
               linewidths=0.4, label="→ nearest train positive")

    ax.axvline(0.5, color="gray", ls=":", lw=0.7, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("SOAP similarity")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    # Custom legend
    _leg = [
        Line2D([], [], color="#a6bddb", lw=6, label="→ any train MOF"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=5, markeredgecolor="k", markeredgewidth=0.3,
               label="→ train positive (high conf.)"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#ff7f0e",
               markersize=5, markeredgecolor="k", markeredgewidth=0.3,
               label="→ train positive (moderate)"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#d62728",
               markersize=5, markeredgecolor="k", markeredgewidth=0.3,
               label="→ train positive (low)"),
    ]
    ax.legend(handles=_leg, loc="lower right", fontsize=5, frameon=True,
              fancybox=False, edgecolor="0.8")
    ax.set_title("Ensemble Predictions: Structural Applicability Domain",
                 fontweight="bold", fontsize=7, pad=6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    _save_panel(fig, output_dir, "analysis_c_applicability")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS D — Representation Agreement (Mantel Test)
# ══════════════════════════════════════════════════════════════════════
def analysis_d_mantel_test(soap_cids, soap_matrix, nn_cids, nn_embs,
                           labeled_bg, n_sample=1000, n_perms=999,
                           seed=42, output_dir="."):
    """Mantel test: permutation-based correlation between SOAP and NN
    pairwise distance matrices.  Reports one number quantifying how
    much the NN captures structural information.
    """

    print("\n  ── Analysis D: Representation Agreement (Mantel Test) ──")

    soap_idx = {c: i for i, c in enumerate(soap_cids)}
    nn_idx   = {c: i for i, c in enumerate(nn_cids)}

    # Common labeled MOFs
    common_labeled = [c for c in soap_cids
                      if c in labeled_bg and c in nn_idx]
    print(f"    Common labeled MOFs: {len(common_labeled)}")

    rng = np.random.RandomState(seed)
    if len(common_labeled) > n_sample:
        common_labeled = list(rng.choice(common_labeled, n_sample,
                                         replace=False))
        print(f"    Subsampled to {n_sample} for tractable computation")

    n = len(common_labeled)
    if n < 10:
        print("    WARNING: Too few common MOFs — skipping Mantel test")
        return None

    # Build distance matrices (1 - similarity)
    s_soap = soap_matrix[[soap_idx[c] for c in common_labeled]]
    s_nn   = nn_embs[[nn_idx[c] for c in common_labeled]]

    soap_dist = 1.0 - soap_similarity_matrix(s_soap)
    nn_dist   = 1.0 - cosine_similarity_matrix(s_nn)

    # Flatten upper triangle
    triu_i, triu_j = np.triu_indices(n, k=1)
    soap_flat = soap_dist[triu_i, triu_j]
    nn_flat   = nn_dist[triu_i, triu_j]

    # Observed correlation
    r_obs, _ = spstats.pearsonr(soap_flat, nn_flat)
    print(f"    Observed Pearson r: {r_obs:.4f}")

    # Permutation test
    print(f"    Running {n_perms} permutations ...")
    r_perms = np.empty(n_perms)
    for k in range(n_perms):
        perm = rng.permutation(n)
        soap_perm = soap_dist[np.ix_(perm, perm)]
        soap_perm_flat = soap_perm[triu_i, triu_j]
        r_perms[k], _ = spstats.pearsonr(soap_perm_flat, nn_flat)

    p_value = (np.sum(r_perms >= r_obs) + 1) / (n_perms + 1)
    print(f"    Mantel r = {r_obs:.4f},  p = {p_value:.4f}")
    print(f"    Permutation r range: [{r_perms.min():.4f}, {r_perms.max():.4f}]")

    # ── Figure: histogram of permuted r values ────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.hist(r_perms, bins=50, color="#a6bddb", edgecolor="white",
            linewidth=0.4, density=True, alpha=0.85)
    ax.axvline(r_obs, color="#e41a1c", lw=2.0, ls="--",
               label=f"Observed r = {r_obs:.3f}")
    ax.set_xlabel("Pearson r (permuted)")
    ax.set_ylabel("Density")
    ax.set_title(f"Mantel Test: SOAP vs. NN Distance Matrices\n"
                 f"r = {r_obs:.3f}, p = {p_value:.4f}  (n = {n})",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    _save_panel(fig, output_dir, "analysis_d_mantel_test")
    plt.close()

    out = {
        "n_mofs": n,
        "n_pairs": len(soap_flat),
        "mantel_r": round(r_obs, 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_perms,
        "perm_r_mean": round(float(r_perms.mean()), 4),
        "perm_r_std": round(float(r_perms.std()), 4),
    }
    jp = os.path.join(output_dir, "analysis_d_mantel.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"    Results -> {jp}")
    return out


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(
        description="SOAP Structural Validation of Data Split",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    pa.add_argument("--cif_dir", required=True,
                    help="Directory with .cif files for all MOFs")
    pa.add_argument("--merged_embeddings", required=True,
                    help="Unified NPZ from forward_pretrained_embeddings.py")
    pa.add_argument("--labeled_splits_dir", required=True,
                    help="Dir with {train,val,test}_bandgaps_regression.json")
    pa.add_argument("--nominations", default=None,
                    help="File with top-K ensemble CIF IDs (one per line) "
                         "for applicability domain analysis")
    pa.add_argument("--output_dir", default="./figures_output/soap_validation")
    pa.add_argument("--threshold", type=float, default=1.0,
                    help="Bandgap threshold in eV (default: 1.0)")
    pa.add_argument("--n_sample_negatives", type=int, default=500,
                    help="Number of negatives to sample for Analysis B")
    pa.add_argument("--mantel_n_sample", type=int, default=1000,
                    help="Max MOFs for Mantel test pairwise matrix")
    pa.add_argument("--mantel_n_perms", type=int, default=999,
                    help="Number of permutations for Mantel test")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--recompute_soap", action="store_true",
                    help="Force re-computation of SOAP descriptors")
    pa.add_argument("--skip_analyses", nargs="*", default=[],
                    choices=["a", "b", "c", "d"],
                    help="Skip specific analyses (e.g., --skip_analyses c d)")

    args = pa.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_publication_style()

    soap_cache_path = os.path.join(args.output_dir, "soap_descriptors.npz")

    print("=" * 70)
    print("  SOAP STRUCTURAL VALIDATION — Independent Split Confirmation")
    print("=" * 70)

    # ── Load metadata ─────────────────────────────────────────────────
    print("\n[0] Loading metadata ...")
    labeled_bg, labeled_splits = load_split_labels(args.labeled_splits_dir)
    labeled_cid_set = set(labeled_bg.keys())

    # Positive counts per split
    for sp in ("train", "val", "test"):
        n_pos = sum(1 for c, s in labeled_splits.items()
                    if s == sp and labeled_bg[c] < args.threshold)
        print(f"      {sp} positives (bg < {args.threshold}): {n_pos}")

    print(f"    Loading NN embeddings: {args.merged_embeddings}")
    nn_data = load_npz_embeddings(args.merged_embeddings)
    nn_cids = nn_data["cif_ids"]
    nn_embs = nn_data["embeddings"]
    print(f"    NN embeddings: {len(nn_cids)} MOFs, dim={nn_embs.shape[1]}")

    # ── STAGE 1: SOAP descriptors ─────────────────────────────────────
    print(f"\n[1/2] SOAP descriptors ...")
    if os.path.exists(soap_cache_path) and not args.recompute_soap:
        print(f"    Cache found: {soap_cache_path}")
        soap_cids, soap_matrix = load_soap_cache(soap_cache_path)
    else:
        print(f"    Computing from CIF files in {args.cif_dir} ...")
        soap_cids, soap_matrix = compute_soap_descriptors(
            args.cif_dir, nn_cids, soap_cache_path)

    soap_set = set(soap_cids)
    n_labeled_in_soap = sum(1 for c in soap_cids if c in labeled_cid_set)
    print(f"\n    SOAP MOFs: {len(soap_cids)}  "
          f"(labeled: {n_labeled_in_soap}, "
          f"unlabeled: {len(soap_cids) - n_labeled_in_soap})")

    # ── STAGE 2: Quantitative analyses ────────────────────────────────
    print(f"\n[2/2] Quantitative analyses ...")
    all_results = {}

    # Analysis A: Split Coverage Validation
    if "a" not in args.skip_analyses:
        all_results["a"] = analysis_a_split_coverage(
            soap_cids, soap_matrix, nn_cids, nn_embs,
            labeled_bg, labeled_splits, args.threshold,
            args.output_dir)

    # Analysis B: Structure–Bandgap Correlation
    if "b" not in args.skip_analyses:
        all_results["b"] = analysis_b_structure_bandgap(
            soap_cids, soap_matrix, labeled_bg, args.threshold,
            args.output_dir,
            n_sample_negatives=args.n_sample_negatives,
            seed=args.seed)

    # Analysis C: Applicability Domain
    if "c" not in args.skip_analyses:
        if args.nominations and os.path.exists(
                args.nominations):
            nomination_cids = load_nominations(
                args.nominations)
            all_results["c"] = analysis_c_applicability_domain(
                soap_cids, soap_matrix, nn_cids, nn_embs,
                labeled_bg, labeled_splits,
                nomination_cids, args.threshold, args.output_dir)
        else:
            print("\n  ── Analysis C: Skipped (no --nominations) ──")

    # Analysis D: Mantel Test
    if "d" not in args.skip_analyses:
        all_results["d"] = analysis_d_mantel_test(
            soap_cids, soap_matrix, nn_cids, nn_embs,
            labeled_bg,
            n_sample=args.mantel_n_sample,
            n_perms=args.mantel_n_perms,
            seed=args.seed,
            output_dir=args.output_dir)

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "soap_mofs": len(soap_cids),
        "nn_mofs": len(nn_cids),
        "labeled_in_soap": n_labeled_in_soap,
        "soap_dim": int(soap_matrix.shape[1]),
        "soap_params": {"r_cut": SOAP_RCUT, "n_max": SOAP_NMAX,
                        "l_max": SOAP_LMAX, "sigma": SOAP_SIGMA},
        "threshold": args.threshold,
    }
    sp = os.path.join(args.output_dir, "soap_validation_summary.json")
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Done.  All outputs in {args.output_dir}/")
    print(f"")
    if "a" in all_results and all_results["a"]:
        a = all_results["a"]["summary"]
        print(f"  A. Coverage: mean SOAP sim = {a['mean_soap_sim']:.3f}, "
              f"mean NN sim = {a['mean_nn_sim']:.3f}")
    if "b" in all_results and all_results["b"]:
        b = all_results["b"]["positive_pairs"]
        print(f"  B. Structure-bandgap: ρ = {b['spearman_rho']:.3f} "
              f"(p = {b['spearman_p']:.1e})")
    if "c" in all_results and all_results["c"]:
        c = all_results["c"]["summary"]
        print(f"  C. Applicability: {c['n_high_confidence']} high, "
              f"{c['n_moderate_confidence']} moderate, "
              f"{c['n_low_confidence']} low confidence")
    if "d" in all_results and all_results["d"]:
        d = all_results["d"]
        print(f"  D. Mantel test: r = {d['mantel_r']:.3f}, "
              f"p = {d['p_value']:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
