"""
Verify the ``qmof_bb_dir/`` and ``qmof_topo_dir/`` artefacts produced by
``scripts/build_custom_dirs.py``.

Loads every topology and every building block through ``pormake.Database``,
prints counts, and reports the QMOF-coverage of the retained whitelists.
Exits with a non-zero status if any file fails to parse.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "qmof_analysis"
DEFAULT_TOPO_OUT = PROJECT_ROOT / "qmof_topo_dir"
DEFAULT_BB_OUT = PROJECT_ROOT / "qmof_bb_dir"


# Elements we treat as "organic filler" inside a node XYZ -- they are legal
# atoms inside a MOF secondary building unit (e.g. formate oxygens around a
# metal cluster) but they are NOT the metal that identifies the node.
_ORGANIC_FILLERS = frozenset({"C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "P", "X"})


def _read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _node_metals(xyz_path: Path) -> set[str]:
    """Return the set of metal-like element symbols in an N*.xyz file."""
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return set()
    try:
        natoms = int(lines[0].strip())
    except ValueError:
        return set()
    metals: set[str] = set()
    for raw in lines[2 : 2 + natoms]:
        parts = raw.split()
        if not parts:
            continue
        el = parts[0]
        if el not in _ORGANIC_FILLERS:
            metals.add(el)
    return metals


def verify(
    analysis_dir: Path,
    topo_dir: Path,
    bb_dir: Path,
) -> int:
    import pormake as pm

    pm.log.disable_print()
    pm.log.disable_file_print()

    cgd_files = sorted(topo_dir.glob("*.cgd"))
    xyz_files = sorted(bb_dir.glob("*.xyz"))
    print(f"[verify] topo dir   = {topo_dir} ({len(cgd_files)} .cgd)")
    print(f"[verify] bb   dir   = {bb_dir} ({len(xyz_files)} .xyz)")

    db = pm.Database(bb_dir=bb_dir, topo_dir=topo_dir)

    # ---- Topologies ----
    good_topo: list[str] = []
    bad_topo: list[tuple[str, str]] = []
    for cgd in cgd_files:
        name = cgd.stem
        try:
            topo = db.get_topo(name)
            _ = topo.unique_cn  # touch an attribute to force parsing
            good_topo.append(name)
        except Exception as exc:
            bad_topo.append((name, repr(exc)))

    # ---- Building blocks ----
    good_bb: list[str] = []
    bad_bb: list[tuple[str, str]] = []
    zero_conn: list[str] = []
    for xyz in xyz_files:
        name = xyz.stem
        try:
            bb = db.get_bb(name)
            n_conn = bb.n_connection_points
            good_bb.append(name)
            if n_conn == 0:
                zero_conn.append(name)
        except Exception as exc:
            bad_bb.append((name, repr(exc)))

    print(f"[verify] topologies loaded: {len(good_topo)}/{len(cgd_files)}")
    if bad_topo:
        print("[verify] topology failures:")
        for n, err in bad_topo[:10]:
            print(f"    - {n}: {err}")
    print(f"[verify] building blocks loaded: {len(good_bb)}/{len(xyz_files)}")
    if bad_bb:
        print("[verify] bb failures:")
        for n, err in bad_bb[:10]:
            print(f"    - {n}: {err}")
    if zero_conn:
        print(f"[verify] WARNING: {len(zero_conn)} BBs have 0 connection points")

    # ---- QMOF coverage sanity check ----
    topo_whitelist = set(_read_list(analysis_dir / "selected_topologies.txt"))
    metal_whitelist = set(_read_list(analysis_dir / "selected_metals.txt"))
    loaded_topo_set = set(good_topo)

    topo_coverage = (
        len(loaded_topo_set & topo_whitelist) / len(topo_whitelist)
        if topo_whitelist else 0.0
    )
    print(
        f"[verify] fraction of selected_topologies successfully loaded: "
        f"{topo_coverage:.1%} ({len(loaded_topo_set & topo_whitelist)}/{len(topo_whitelist)})"
    )
    print(f"[verify] metal whitelist size: {len(metal_whitelist)}")

    # ---- Metal-distribution diagnostic ----
    # This counts how many node XYZ files CONTAIN each metal. Under PORMAKE's
    # uniform-over-files sampling, a candidate node has probability roughly
    # proportional to that count. Use this to decide whether you need the
    # reweighting step in scripts/reweight_candidates.py.
    node_xyzs = [p for p in xyz_files if p.stem.startswith("N")]
    per_metal_nodes: Counter[str] = Counter()
    nodes_with_metals = 0
    for path in node_xyzs:
        metals = _node_metals(path)
        if metals:
            nodes_with_metals += 1
        for m in metals:
            per_metal_nodes[m] += 1

    print(
        f"[verify] metal distribution across {len(node_xyzs)} node XYZs "
        f"({nodes_with_metals} contain at least one metal):"
    )
    if node_xyzs:
        width = max((len(m) for m in per_metal_nodes), default=2)
        total = len(node_xyzs)
        for metal, count in per_metal_nodes.most_common():
            bar = "#" * min(40, int(40 * count / max(per_metal_nodes.values())))
            share = count / total
            print(
                f"    {metal:<{width}s}  {count:>4d}  {share:>6.1%}  {bar}"
            )
    alkali_like = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba"}
    alkali_total = sum(per_metal_nodes[m] for m in alkali_like)
    if alkali_total:
        print(
            f"[verify] note: {alkali_total}/{len(node_xyzs)} "
            f"({alkali_total / max(1, len(node_xyzs)):.1%}) node XYZs contain "
            "alkali/alkaline-earth metals; consider reweighting if your "
            "application favors transition-metal MOFs."
        )

    exit_code = 0
    if bad_topo or bad_bb:
        exit_code = 1
    if not good_topo or not good_bb:
        exit_code = 1
    return exit_code


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--topo-dir", type=Path, default=DEFAULT_TOPO_OUT)
    p.add_argument("--bb-dir", type=Path, default=DEFAULT_BB_OUT)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(verify(args.analysis_dir, args.topo_dir, args.bb_dir))
