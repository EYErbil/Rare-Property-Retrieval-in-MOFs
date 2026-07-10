"""
Chemistry-aware reweighting of PORMAKE candidate lists.

PORMAKE's ``make_candidates.py`` samples nodes/edges/topologies uniformly
over files. That means, e.g., Na-containing MOFs and Cu-containing MOFs get
probability proportional to how many node XYZs contain each metal -- not to
how useful each metal is for your application.

This script takes a *larger-than-needed* candidate list (3-5x oversample) and
weighted-subsamples it down to the target size, using a priority profile over
metals (and optionally topologies). No upstream code is modified.

Typical workflow
----------------
    # 1. Generate ~4x the desired number of candidates
    python bulk_pormake_generation/make_candidates.py -n 40000 \
        --bb-dir qmof_bb_dir --topo-dir qmof_topo_dir \
        --pre-defined-list qmof_analysis/qmof_rmsd_nodes.pickle \
        --has-metal=True --save qmof_candidates.txt

    # 2. Reweight to 10 000 with the built-in "conductor" profile
    python scripts/reweight_candidates.py -n 10000 \
        --candidates qmof_candidates.txt \
        --bb-dir    qmof_bb_dir \
        --profile   conductor \
        --save      qmof_candidates_weighted.txt

    # 3. Feed the filtered list to build_materials.py as usual

Profiles
--------
- ``conductor`` (default): favor Cu, Ni, Ag, Co, Fe, Mn, V, Pd, Pt, Ru;
  disfavor alkali, alkaline-earth, Al, lanthanides.
- ``qmof_match``: uniform over metals (i.e. preserves PORMAKE's own per-file
  uniform distribution; effectively a random subsample).
- Custom: ``--profile-json path/to/profile.json`` with the same shape as
  ``CONDUCTOR_PROFILE`` below.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BB_DIR = PROJECT_ROOT / "qmof_bb_dir"
DEFAULT_CANDIDATES = PROJECT_ROOT / "qmof_candidates.txt"
DEFAULT_SAVE = PROJECT_ROOT / "qmof_candidates_weighted.txt"

_ORGANIC_FILLERS = frozenset({"C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "P", "X"})


# ---------------------------------------------------------------------------
# Built-in priority profiles
# ---------------------------------------------------------------------------

CONDUCTOR_PROFILE: dict = {
    "metals": {
        # Strongly favored -- conductive / redox-active transition metals
        "Cu": 3.0, "Ni": 3.0, "Ag": 2.5, "Co": 2.5, "Fe": 2.5,
        "Ru": 2.0, "Mn": 1.8, "V": 1.8, "Cr": 1.5, "Mo": 1.5,
        "Rh": 1.5, "Pd": 1.5, "Pt": 1.5, "W":  1.3, "Ir": 1.3,
        # Neutral baseline -- classic MOF metals, not specifically conductive
        "Ti": 1.2, "Zn": 1.0, "Zr": 1.0, "Hf": 0.9, "Sn": 0.8,
        # Lanthanides -- expensive, less studied for conductivity
        "La": 0.3, "Ce": 0.4, "Pr": 0.3, "Nd": 0.3, "Sm": 0.3,
        "Eu": 0.4, "Gd": 0.4, "Tb": 0.3, "Dy": 0.3, "Ho": 0.3,
        "Er": 0.3, "Tm": 0.2, "Yb": 0.3, "Lu": 0.2,
        # Main-group -- generally insulating
        "Al": 0.3, "Ga": 0.3, "In": 0.4, "Tl": 0.1,
        "B":  0.3, "Si": 0.3, "Ge": 0.3, "As": 0.2, "Sb": 0.2, "Bi": 0.3,
        # Disfavored -- alkali & alkaline-earth form insulating MOFs
        "Li": 0.10, "Na": 0.05, "K":  0.05, "Rb": 0.02, "Cs": 0.02,
        "Be": 0.10, "Mg": 0.20, "Ca": 0.20, "Sr": 0.10, "Ba": 0.10,
        # Actinides -- radioactive, skip
        "Th": 0.05, "U":  0.05,
    },
    "default_metal": 1.0,
    "topologies": {},
    "default_topology": 1.0,
    # Floor to avoid exact-zero probability so that even disfavored metals are
    # occasionally sampled (a hard zero would make extreme-tail exclusions too
    # brittle to single-atom metal impurities in multi-metal nodes).
    "min_weight": 1e-3,
}


QMOF_MATCH_PROFILE: dict = {
    "metals": {},
    "default_metal": 1.0,
    "topologies": {},
    "default_topology": 1.0,
    "min_weight": 1e-3,
}


BUILTIN_PROFILES: dict[str, dict] = {
    "conductor":  CONDUCTOR_PROFILE,
    "qmof_match": QMOF_MATCH_PROFILE,
}


# ---------------------------------------------------------------------------
# XYZ reading (metal extraction)
# ---------------------------------------------------------------------------

def _read_xyz_elements(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return []
    try:
        natoms = int(lines[0].strip())
    except ValueError:
        return []
    out: list[str] = []
    for raw in lines[2 : 2 + natoms]:
        parts = raw.split()
        if not parts:
            continue
        out.append(parts[0])
    return out


def _metals_in_node(xyz_path: Path) -> set[str]:
    return {
        el for el in _read_xyz_elements(xyz_path)
        if el not in _ORGANIC_FILLERS and el != "X"
    }


def _load_node_metal_index(bb_dir: Path) -> dict[str, set[str]]:
    """Map each node name (stem of N*.xyz) to its set of metal elements."""
    index: dict[str, set[str]] = {}
    for xyz in bb_dir.glob("N*.xyz"):
        index[xyz.stem] = _metals_in_node(xyz)
    return index


# ---------------------------------------------------------------------------
# Candidate parsing
# ---------------------------------------------------------------------------

def _parse_candidate(line: str) -> tuple[str, list[str], list[str]]:
    """Split a candidate name into (topology, node_names, edge_names).

    Candidate format (from PORMAKE's ``make_mof_name``):
        "<topo_name>+<node1>+<node2>+...+<edge1>+<edge2>+..."
    Node names start with 'N', edge names start with 'E' or 'L', sentinel "E0"
    means "no edge at that slot".
    """
    parts = [p for p in line.strip().split("+") if p]
    if not parts:
        return "", [], []
    topo = parts[0]
    nodes: list[str] = []
    edges: list[str] = []
    for p in parts[1:]:
        if p.startswith("N"):
            nodes.append(p)
        elif p.startswith(("E", "L")):
            edges.append(p)
        else:
            # Unknown prefix -- treat as node by convention (shouldn't happen).
            nodes.append(p)
    return topo, nodes, edges


# ---------------------------------------------------------------------------
# Scoring + sampling
# ---------------------------------------------------------------------------

def _score_candidate(
    topo: str,
    metals: Iterable[str],
    profile: dict,
) -> float:
    metal_w = profile.get("metals", {})
    topo_w = profile.get("topologies", {})
    default_m = float(profile.get("default_metal", 1.0))
    default_t = float(profile.get("default_topology", 1.0))
    floor = float(profile.get("min_weight", 1e-3))

    score = float(topo_w.get(topo, default_t))
    for m in metals:
        w = float(metal_w.get(m, default_m))
        score *= max(w, floor)
    return max(score, floor)


def _weighted_sample_without_replacement(
    scores: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Efraimidis-Spirakis weighted reservoir sampling.

    Stable for small weights because it operates on log-keys internally.
    Returns the indices of the ``n`` sampled items.
    """
    if n >= len(scores):
        return np.arange(len(scores))
    u = rng.random(len(scores))
    log_keys = np.log(u) / np.maximum(scores, 1e-300)
    # Largest log-key wins; argpartition gives the top-n indices.
    idx = np.argpartition(-log_keys, n - 1)[:n]
    return idx


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def reweight(
    candidates_path: Path,
    bb_dir: Path,
    profile: dict,
    target_n: int,
    save_path: Path,
    *,
    seed: int = 42,
) -> None:
    lines = [
        ln.strip()
        for ln in candidates_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not lines:
        raise ValueError(f"No candidates found in {candidates_path}")

    node_index = _load_node_metal_index(bb_dir)
    if not node_index:
        raise ValueError(f"No N*.xyz files found in {bb_dir}")

    print(f"[reweight] loaded {len(lines)} candidates, {len(node_index)} known nodes")

    scores = np.empty(len(lines), dtype=np.float64)
    kept_metals_tally: Counter[str] = Counter()
    missing_nodes = 0
    for i, line in enumerate(lines):
        topo, nodes, _edges = _parse_candidate(line)
        metals: set[str] = set()
        for n in nodes:
            if n in node_index:
                metals |= node_index[n]
            else:
                missing_nodes += 1
        scores[i] = _score_candidate(topo, metals, profile)
        for m in metals:
            kept_metals_tally[m] += 1

    if missing_nodes:
        print(
            f"[reweight] warning: {missing_nodes} node references did not match "
            f"any N*.xyz in {bb_dir}; those candidates scored only from topology"
        )

    rng = np.random.default_rng(seed)
    n = min(target_n, len(lines))
    picks = _weighted_sample_without_replacement(scores, n, rng)

    # Preserve original order (make results easier to diff / reproduce).
    picks = np.sort(picks)
    sampled_lines = [lines[i] for i in picks]

    save_path.write_text("\n".join(sampled_lines) + "\n", encoding="utf-8")
    print(f"[reweight] wrote {len(sampled_lines)} candidates -> {save_path}")

    # Post-sample metal distribution report
    post_metals: Counter[str] = Counter()
    for i in picks:
        _, nodes, _ = _parse_candidate(lines[i])
        for n_name in nodes:
            for m in node_index.get(n_name, set()):
                post_metals[m] += 1

    print("[reweight] metal distribution across sampled candidates (top 20):")
    total_post = sum(post_metals.values()) or 1
    for metal, count in post_metals.most_common(20):
        share = count / total_post
        pre_share = kept_metals_tally[metal] / max(1, sum(kept_metals_tally.values()))
        delta = share - pre_share
        arrow = "+" if delta >= 0 else "-"
        print(
            f"    {metal:<3s}  {count:>6d}  {share:>6.1%}  "
            f"(input {pre_share:>6.1%}, delta {arrow}{abs(delta):.1%})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_profile(name_or_path: str) -> dict:
    if name_or_path in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name_or_path]
    p = Path(name_or_path)
    if not p.is_file():
        raise SystemExit(
            f"Unknown profile '{name_or_path}' (not a built-in and not a file). "
            f"Built-ins: {list(BUILTIN_PROFILES)}"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--candidates", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("-b", "--bb-dir", type=Path, default=DEFAULT_BB_DIR)
    p.add_argument("-n", "--target-n", type=int, default=10000)
    p.add_argument("-s", "--save", type=Path, default=DEFAULT_SAVE)
    p.add_argument(
        "-p", "--profile",
        default="conductor",
        help=f"Built-in profile name or path to a JSON file. Built-ins: {list(BUILTIN_PROFILES)}",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    profile = _load_profile(args.profile)
    reweight(
        args.candidates,
        args.bb_dir,
        profile,
        args.target_n,
        args.save,
        seed=args.seed,
    )
    sys.exit(0)
