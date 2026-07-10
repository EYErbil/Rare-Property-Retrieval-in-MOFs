"""
Build QMOF-restricted ``bb-dir`` and ``topo-dir`` for PORMAKE.

Inputs (from ``scripts/analyze_reference_db.py``):
- ``qmof_analysis/selected_topologies.txt``
- ``qmof_analysis/selected_metals.txt``
- ``qmof_analysis/selected_linkers.txt``

Outputs:
- ``qmof_topo_dir/``  -- ``.cgd`` topology files copied from PORMAKE's default
  topology database for every code in ``selected_topologies.txt``.
- ``qmof_bb_dir/``    -- PORMAKE default node XYZs filtered by metal membership,
  all default edge XYZs, and optionally a handful of custom linker XYZs
  generated from SMILES via RDKit.
- ``qmof_analysis/build_log.md`` -- human-readable summary of what was copied,
  skipped, or failed.

Run order:
    python scripts/analyze_reference_db.py
    python scripts/build_custom_dirs.py           # all steps
    python scripts/build_custom_dirs.py --no-augment   # skip RDKit linker gen
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "qmof_analysis"
DEFAULT_TOPO_OUT = PROJECT_ROOT / "qmof_topo_dir"
DEFAULT_BB_OUT = PROJECT_ROOT / "qmof_bb_dir"


# ---------------------------------------------------------------------------
# PORMAKE default database discovery
# ---------------------------------------------------------------------------

def _pormake_default_dirs() -> tuple[Path, Path]:
    """Locate PORMAKE's default ``bb_dir`` and ``topo_dir``.

    We prefer the public attributes on ``pm.Database()``; if those are missing
    in the installed version, we fall back to the package's ``database``
    subfolder.
    """
    import pormake as pm

    db = pm.Database()
    bb_dir = Path(getattr(db, "bb_dir", "") or "")
    topo_dir = Path(getattr(db, "topo_dir", "") or "")

    if not bb_dir.is_dir() or not topo_dir.is_dir():
        pkg_root = Path(pm.__file__).resolve().parent
        for candidate in (pkg_root / "database", pkg_root / "data"):
            if (candidate / "bbs").is_dir() and not bb_dir.is_dir():
                bb_dir = candidate / "bbs"
            if (candidate / "topologies").is_dir() and not topo_dir.is_dir():
                topo_dir = candidate / "topologies"

    if not bb_dir.is_dir():
        raise RuntimeError(
            "Could not locate PORMAKE's default bb_dir; inspected "
            f"{bb_dir!s}"
        )
    if not topo_dir.is_dir():
        raise RuntimeError(
            "Could not locate PORMAKE's default topo_dir; inspected "
            f"{topo_dir!s}"
        )
    return bb_dir, topo_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/analyze_reference_db.py first"
        )
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_xyz_elements(xyz_path: Path) -> list[str]:
    """Return the element column of an XYZ file."""
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return []
    try:
        natoms = int(lines[0].strip())
    except ValueError:
        return []
    elements: list[str] = []
    for raw in lines[2 : 2 + natoms]:
        parts = raw.split()
        if not parts:
            continue
        elements.append(parts[0])
    return elements


# ---------------------------------------------------------------------------
# Step 2a -- topo-dir
# ---------------------------------------------------------------------------

def build_topo_dir(
    selected_topos: list[str],
    default_topo_dir: Path,
    out_dir: Path,
) -> dict[str, list[str]]:
    """Copy ``.cgd`` files for every selected topology code into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for code in selected_topos:
        src = default_topo_dir / f"{code}.cgd"
        if src.is_file():
            shutil.copy2(src, out_dir / src.name)
            copied.append(code)
        else:
            missing.append(code)
    print(f"[topo] copied {len(copied)}/{len(selected_topos)} topologies -> {out_dir}")
    if missing:
        print(f"[topo] missing from PORMAKE defaults: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return {"copied": copied, "missing": missing}


# ---------------------------------------------------------------------------
# Step 2b.1 / 2b.2 -- filter default nodes + keep all default edges
# ---------------------------------------------------------------------------

def filter_and_copy_bbs(
    selected_metals: set[str],
    default_bb_dir: Path,
    out_dir: Path,
) -> dict[str, list[str]]:
    """Copy default node XYZs (``N*``) whose metal atoms are all in
    ``selected_metals``, plus all default edge XYZs (``E*``/``L*``)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_nodes: list[str] = []
    skipped_nodes: list[str] = []
    kept_edges: list[str] = []
    odd_bbs: list[str] = []

    # Elements we expect to find in organic linker parts of nodes (non-metals that
    # are fine to keep alongside a metal).
    _organic_fillers = {"C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "P", "X"}

    for xyz in sorted(default_bb_dir.glob("*.xyz")):
        name = xyz.stem
        elements = _read_xyz_elements(xyz)
        if not elements:
            odd_bbs.append(name)
            continue
        non_dummy = [e for e in elements if e != "X"]
        metal_elements = {e for e in non_dummy if e not in _organic_fillers}

        if name.startswith("N"):
            if not metal_elements:
                skipped_nodes.append(name)
                continue
            if metal_elements.issubset(selected_metals):
                shutil.copy2(xyz, out_dir / xyz.name)
                kept_nodes.append(name)
            else:
                skipped_nodes.append(name)
        elif name.startswith(("E", "L")):
            shutil.copy2(xyz, out_dir / xyz.name)
            kept_edges.append(name)
        else:
            odd_bbs.append(name)
            shutil.copy2(xyz, out_dir / xyz.name)

    print(
        f"[bbs] kept {len(kept_nodes)} metal nodes, "
        f"{len(kept_edges)} organic edges; skipped {len(skipped_nodes)} nodes"
    )
    if odd_bbs:
        print(f"[bbs] unclassified BB names copied as-is: {odd_bbs[:10]}{' ...' if len(odd_bbs) > 10 else ''}")
    return {
        "kept_nodes": kept_nodes,
        "skipped_nodes": skipped_nodes,
        "kept_edges": kept_edges,
        "odd_bbs": odd_bbs,
    }


# ---------------------------------------------------------------------------
# Step 2b.3 -- optional linker augmentation from SMILES
# ---------------------------------------------------------------------------

# SMARTS patterns for QMOF-style "leaving" functional groups. For each match:
#   * The whole matched fragment is DELETED from the molecule.
#   * A single dummy ``X`` atom is placed at the position of the anchor atom
#     (the first atom in the SMARTS -- flagged with ``:1``).
# This matches PORMAKE's XYZ convention where an edge linker has its connection
# chemistry represented purely by an ``X`` atom bonded (via distance) to the
# ring/skeleton carbon it was attached to.
CONNECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    # -C(=O)O- / -C(=O)OH : delete C and both O atoms, X at C position
    ("carboxylate", "[CX3:1](=O)[OX1H0-,OX2H1]"),
    # -P(=O)(O)(O) : delete P and all three O atoms, X at P position
    ("phosphonate", "[PX4:1](=O)([OX1H0-,OX2H1])[OX1H0-,OX2H1]"),
    # -S(=O)(=O)O : delete S and all three O atoms, X at S position
    ("sulfonate",   "[SX4:1](=O)(=O)[OX1H0-,OX2H1]"),
)


def _embed_3d(smiles: str):
    """Return an RDKit mol with a 3D conformer, or ``None`` on failure."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover -- RDKit optional
        raise RuntimeError("RDKit is required for linker augmentation") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:  # pragma: no cover -- fall through with unoptimized conformer
        pass
    return mol


def _find_leaving_groups(mol):
    """Identify leaving functional groups, X positions, and each X's anchor.

    Returns ``(delete_indices, connection_sites)`` where ``connection_sites`` is
    a list of ``(anchor_neighbor_idx, (x, y, z))`` tuples. ``anchor_neighbor_idx``
    is the index (in the original mol) of the surviving heavy atom that the
    new dummy ``X`` atom should bond to -- typically the ring/skeleton carbon
    that was attached to the excised carboxyl carbon.
    """
    from rdkit import Chem

    if mol.GetNumConformers() == 0:
        return [], []
    conf = mol.GetConformer(0)

    delete: set[int] = set()
    claimed_anchors: set[int] = set()
    raw_sites: list[tuple[int, int]] = []  # (anchor_atom_idx, match_tuple_index)
    anchor_to_pos: dict[int, tuple[float, float, float]] = {}

    for _name, smarts in CONNECTION_PATTERNS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            anchor = match[0]
            if anchor in claimed_anchors:
                continue
            claimed_anchors.add(anchor)
            delete.update(match)
            pos = conf.GetAtomPosition(anchor)
            anchor_to_pos[anchor] = (pos.x, pos.y, pos.z)
            raw_sites.append((anchor, anchor))  # placeholder; resolved below

    # Drop H atoms whose only heavy neighbor is already slated for removal.
    extra_h: set[int] = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        heavy_neighbors = [
            n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() != 1
        ]
        if heavy_neighbors and all(n in delete for n in heavy_neighbors):
            extra_h.add(atom.GetIdx())
    delete |= extra_h

    # For every claimed anchor, the X atom should bond to the anchor's surviving
    # heavy neighbor (the "skeleton" atom the leaving group was hanging off of).
    connection_sites: list[tuple[int, tuple[float, float, float]]] = []
    for anchor, _ in raw_sites:
        atom = mol.GetAtomWithIdx(anchor)
        surviving = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetIdx() not in delete and n.GetAtomicNum() != 1
        ]
        if not surviving:
            # Degenerate: no skeleton atom for X to bond to (e.g. formate).
            continue
        connection_sites.append((surviving[0], anchor_to_pos[anchor]))

    return sorted(delete), connection_sites


_BOND_TYPE_LETTER = {
    "SINGLE": "S",
    "DOUBLE": "D",
    "TRIPLE": "T",
    "AROMATIC": "A",
}


def _linker_to_xyz_text(
    mol,
    delete_indices: list[int],
    connection_sites: list[tuple[int, tuple[float, float, float]]],
    name: str,
) -> str | None:
    """Serialize the mol in PORMAKE's XYZ-with-bond-table format.

    * Atoms in ``delete_indices`` are removed.
    * Remaining atoms are re-indexed densely (0, 1, 2, ...).
    * One dummy ``X`` atom is appended per entry in ``connection_sites`` at the
      given position, bonded (single) to the ``anchor_neighbor`` atom.
    * All RDKit bonds between two surviving atoms are emitted with their bond
      type letter (S / D / T / A).

    Returns ``None`` if fewer than two connection sites would be written.
    """
    if mol.GetNumConformers() == 0:
        return None
    if len(connection_sites) < 2:
        return None

    conf = mol.GetConformer(0)
    delete_set = set(delete_indices)
    natoms = mol.GetNumAtoms()

    # Build old-idx -> new-idx map in stable order.
    new_index: dict[int, int] = {}
    atom_lines: list[str] = []
    for old in range(natoms):
        if old in delete_set:
            continue
        new = len(new_index)
        new_index[old] = new
        atom = mol.GetAtomWithIdx(old)
        pos = conf.GetAtomPosition(old)
        atom_lines.append(
            f"{atom.GetSymbol():<2s} {pos.x: .6f} {pos.y: .6f} {pos.z: .6f}"
        )

    # Append X atoms and record their new indices + anchor neighbors.
    x_new_indices: list[tuple[int, int]] = []  # (x_new_idx, anchor_neighbor_new_idx)
    for anchor_neighbor_old, (x, y, z) in connection_sites:
        if anchor_neighbor_old not in new_index:
            return None
        x_new = len(new_index) + len(x_new_indices)
        x_new_indices.append((x_new, new_index[anchor_neighbor_old]))
        atom_lines.append(f"{'X':<2s} {x: .6f} {y: .6f} {z: .6f}")

    total_atoms = len(atom_lines)

    # Bond table: all mol-internal bonds whose endpoints both survived, plus
    # one single bond per X atom to its anchor neighbor.
    bond_lines: list[str] = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in delete_set or b in delete_set:
            continue
        bt = str(bond.GetBondType()).split(".")[-1]
        letter = _BOND_TYPE_LETTER.get(bt, "S")
        bond_lines.append(
            f"{new_index[a]:<8d}  {new_index[b]:<8d}  {letter}"
        )
    for x_new, anchor_new in x_new_indices:
        bond_lines.append(
            f"{anchor_new:<8d}  {x_new:<8d}  S"
        )

    header = f"{total_atoms}\n{name} (custom QMOF linker, {len(connection_sites)} X)\n"
    return header + "\n".join(atom_lines + bond_lines) + "\n"


def augment_with_qmof_linkers(
    selected_linkers: Iterable[str],
    out_dir: Path,
    log_dir: Path,
    *,
    max_adds: int = 20,
    name_prefix: str = "E9",
) -> dict[str, list[str]]:
    """Try to generate XYZ linkers for QMOF SMILES not already in PORMAKE.

    We do NOT attempt strict duplicate detection against the default library --
    that would require matching an RDKit fragment to a XYZ file. Instead we
    write up to ``max_adds`` novel linkers with stable names ``E9000``,
    ``E9001``, ... and leave PORMAKE to do the geometric matching during the
    RMSD step. If any name already exists in ``out_dir`` we skip it.
    """
    try:
        import rdkit  # noqa: F401
    except ImportError:
        print("[augment] RDKit not installed -- skipping linker augmentation")
        return {"added": [], "failed": [], "skipped": []}

    added: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Purge any stale augmented linkers from a previous run so we don't end up
    # with leftover E9xxx files whose numbering no longer matches the current
    # augmentation attempt.
    for stale in out_dir.glob(f"{name_prefix}*.xyz"):
        stale.unlink()

    idx = 0
    for smiles in selected_linkers:
        if len(added) >= max_adds:
            break
        name = f"{name_prefix}{idx:03d}"
        idx += 1
        target = out_dir / f"{name}.xyz"
        if target.exists():
            skipped.append((name, smiles))
            continue
        mol = _embed_3d(smiles)
        if mol is None:
            failed.append((name, f"{smiles} -- embed failed"))
            continue
        delete_idx, conn_sites = _find_leaving_groups(mol)
        if len(conn_sites) < 2:
            failed.append(
                (name, f"{smiles} -- only {len(conn_sites)} connection point(s)")
            )
            continue
        xyz_text = _linker_to_xyz_text(mol, delete_idx, conn_sites, name)
        if xyz_text is None:
            failed.append((name, f"{smiles} -- xyz serialization failed"))
            continue
        target.write_text(xyz_text, encoding="utf-8")
        added.append((name, smiles))

    log_path = log_dir / "linker_augment_log.md"
    lines = ["# Linker augmentation log", ""]
    lines.append(f"- Attempted up to {max_adds} additions")
    lines.append(f"- Added: {len(added)}")
    lines.append(f"- Failed: {len(failed)}")
    lines.append(f"- Skipped (name collision): {len(skipped)}")
    lines.append("")
    lines.append("## Added")
    for name, smi in added:
        lines.append(f"- `{name}` : `{smi}`")
    lines.append("")
    lines.append("## Failed")
    for name, msg in failed:
        lines.append(f"- `{name}` : {msg}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[augment] added {len(added)} linkers, {len(failed)} failed, {len(skipped)} skipped")
    return {
        "added": [n for n, _ in added],
        "failed": [n for n, _ in failed],
        "skipped": [n for n, _ in skipped],
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_all(
    analysis_dir: Path,
    topo_out: Path,
    bb_out: Path,
    *,
    do_augment: bool = True,
    max_augment: int = 20,
) -> None:
    selected_topos = _read_list(analysis_dir / "selected_topologies.txt")
    selected_metals = set(_read_list(analysis_dir / "selected_metals.txt"))
    selected_linkers = _read_list(analysis_dir / "selected_linkers.txt")

    default_bb_dir, default_topo_dir = _pormake_default_dirs()
    print(f"[paths] PORMAKE default bb_dir  = {default_bb_dir}")
    print(f"[paths] PORMAKE default topo_dir = {default_topo_dir}")

    topo_res = build_topo_dir(selected_topos, default_topo_dir, topo_out)
    bb_res = filter_and_copy_bbs(selected_metals, default_bb_dir, bb_out)

    if do_augment:
        aug_res = augment_with_qmof_linkers(
            selected_linkers, bb_out, analysis_dir, max_adds=max_augment
        )
    else:
        aug_res = {"added": [], "failed": [], "skipped": []}

    log_path = analysis_dir / "build_log.md"
    out_lines = [
        "# QMOF bb-dir / topo-dir build log",
        "",
        f"- topo_out = `{topo_out}`",
        f"- bb_out   = `{bb_out}`",
        "",
        "## Topology copy",
        f"- Requested: {len(selected_topos)}",
        f"- Copied:    {len(topo_res['copied'])}",
        f"- Missing from PORMAKE: {len(topo_res['missing'])}",
        "",
        "### Missing topology codes",
        *(f"- `{code}`" for code in topo_res["missing"]),
        "",
        "## Building-block filtering",
        f"- Kept metal nodes:  {len(bb_res['kept_nodes'])}",
        f"- Skipped nodes (metal not whitelisted / no metal): {len(bb_res['skipped_nodes'])}",
        f"- Kept organic edges: {len(bb_res['kept_edges'])}",
        "",
        "## Linker augmentation",
        f"- Added: {len(aug_res['added'])}",
        f"- Failed: {len(aug_res['failed'])}",
        f"- Skipped: {len(aug_res['skipped'])}",
    ]
    log_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {log_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--topo-out", type=Path, default=DEFAULT_TOPO_OUT)
    p.add_argument("--bb-out", type=Path, default=DEFAULT_BB_OUT)
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip the optional RDKit-based linker augmentation step",
    )
    p.add_argument(
        "--max-augment",
        type=int,
        default=20,
        help="Upper bound on number of custom linkers to add (default: 20)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_all(
        args.analysis_dir,
        args.topo_out,
        args.bb_out,
        do_augment=not args.no_augment,
        max_augment=args.max_augment,
    )
