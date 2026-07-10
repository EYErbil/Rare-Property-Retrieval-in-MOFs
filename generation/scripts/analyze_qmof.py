"""
Reference-dataset distribution analysis (QMOF by default).

Parses a reference MOF table (``qmof.csv`` by default) and emits frequency
tables for topologies, metal elements and canonical linker SMILES, plus a
summary of pore sizes (PLD / LCD). The three ``selected_*.txt`` whitelists it
writes define the generation chemistry consumed by
``scripts/build_custom_dirs.py``.

Use your OWN dataset by pointing ``--csv`` at any CSV and mapping its column
names with ``--topology-col`` / ``--nodes-col`` / ``--linkers-col`` (and
optionally ``--pld-col`` / ``--lcd-col``). The defaults are QMOF's MOFid
columns, so a plain ``python scripts/analyze_qmof.py`` reproduces the paper.
Expected content per row:

- topology column: one or more RCSR-style net codes (comma-separated)
- nodes column:    a Python-list literal of node SMILES with bracketed metals,
  e.g. ``"['[Zn]', '[Cu]']"``
- linkers column:  a Python-list literal of organic linker SMILES

Outputs land in ``--out`` (``qmof_analysis/`` by default):

- ``topology_counts.csv`` / ``metal_counts.csv`` / ``linker_counts.csv``
- ``pld_lcd_summary.json``
- ``selected_topologies.txt`` / ``selected_metals.txt`` / ``selected_linkers.txt``
- ``coverage_report.md``
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CSV = PROJECT_ROOT / "qmof.csv"
DEFAULT_OUT = PROJECT_ROOT / "qmof_analysis"


METALS: frozenset[str] = frozenset(
    {
        # Alkali
        "Li", "Na", "K", "Rb", "Cs", "Fr",
        # Alkaline earth
        "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
        # Transition metals (d-block)
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
        # Lanthanides
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu",
        # Actinides
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
        "Es", "Fm", "Md", "No", "Lr",
        # Post-transition (commonly called metals)
        "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Po",
        # Metalloids sometimes appearing as "metal" nodes in MOFid
        "B", "Si", "Ge", "As", "Sb", "Te",
    }
)


ELEMENT_TOKEN = re.compile(r"\[([A-Z][a-z]?)(?:[@+\-0-9H]*)\]")


def _parse_list_cell(cell: object) -> list[str]:
    """Parse cells like ``"['[Zn]', '[Cu]']"`` into a Python list of strings.

    Returns an empty list for blanks / malformed values.
    """
    if cell is None:
        return []
    if isinstance(cell, float):  # NaN slips in as float
        return []
    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, (list, tuple)):
        return [str(x) for x in parsed]
    return [str(parsed)]


# QMOF sometimes emits non-topology sentinels in the topology column when MOFid
# fails -- strip those so they never reach the whitelist.
_TOPOLOGY_SENTINELS = {"ERROR", "UNKNOWN", "NA", "NONE"}


def _split_topology_cell(cell: object) -> list[str]:
    if cell is None:
        return []
    if isinstance(cell, float):
        return []
    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return []
    codes = [part.strip() for part in text.split(",") if part.strip()]
    return [c for c in codes if c.upper() not in _TOPOLOGY_SENTINELS]


def _extract_elements(smiles: str) -> list[str]:
    """Pull bracketed element symbols out of a SMILES string."""
    return ELEMENT_TOKEN.findall(smiles)


def _canonicalize_smiles(smiles: str) -> str | None:
    """Return RDKit-canonicalized SMILES, or ``None`` if RDKit can't parse it.

    Falls back to the raw (stripped) string when RDKit is unavailable so the
    pipeline still runs without optional deps.
    """
    smiles = smiles.strip()
    if not smiles:
        return None
    try:
        from rdkit import Chem  # type: ignore
        from rdkit import RDLogger  # type: ignore

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except ImportError:
        return smiles


def _select_by_coverage(
    counts: Counter, total_rows: int, target_fraction: float
) -> tuple[list[str], float]:
    """Return the shortest prefix of most-frequent items covering ``target_fraction`` of ``total_rows``.

    Coverage here is ``rows_with_at_least_one_item_in_whitelist / total_rows``.
    Because multi-labeled rows exist, we cannot simply use cumulative counts;
    instead this caller already passed counts keyed by item, and we take items
    in descending count order until the cumulative item count hits the target.
    """
    if total_rows == 0:
        return [], 0.0
    ordered = counts.most_common()
    target = target_fraction * sum(counts.values())
    selected: list[str] = []
    running = 0
    for item, n in ordered:
        selected.append(item)
        running += n
        if running >= target:
            break
    achieved = running / sum(counts.values()) if counts else 0.0
    return selected, achieved


def _full_row_coverage(items_per_row: Iterable[set[str]], whitelist: set[str]) -> float:
    """Fraction of rows whose entire item set is contained in ``whitelist``."""
    total = 0
    covered = 0
    for row_items in items_per_row:
        total += 1
        if not row_items:
            continue
        if row_items.issubset(whitelist):
            covered += 1
    return covered / total if total else 0.0


def analyze(
    csv_path: Path,
    out_dir: Path,
    *,
    topo_coverage: float = 0.95,
    metal_coverage: float = 0.99,
    linker_top_n: int = 200,
    topology_col: str = "info.mofid.topology",
    nodes_col: str = "info.mofid.smiles_nodes",
    linkers_col: str = "info.mofid.smiles_linkers",
    pld_col: str = "info.pld",
    lcd_col: str = "info.lcd",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze] reading {csv_path}")
    # Read the header first so we can (a) validate the required columns and
    # (b) treat the pore-geometry columns as optional for non-QMOF datasets.
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    required = {
        "topology": topology_col,
        "nodes": nodes_col,
        "linkers": linkers_col,
    }
    missing = {role: col for role, col in required.items() if col not in header}
    if missing:
        raise SystemExit(
            f"[analyze] required column(s) not found in {csv_path}:\n"
            + "\n".join(f"    --{role}-col '{col}'" for role, col in missing.items())
            + "\n[analyze] columns present include: "
            + ", ".join(header[:12])
            + (" ..." if len(header) > 12 else "")
            + "\n[analyze] pass the matching --topology-col / --nodes-col / "
            "--linkers-col for your file."
        )
    has_pld = pld_col in header
    has_lcd = lcd_col in header
    usecols = [topology_col, nodes_col, linkers_col]
    usecols += [pld_col] if has_pld else []
    usecols += [lcd_col] if has_lcd else []
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    n_rows = len(df)
    print(f"[analyze] loaded {n_rows} rows")

    # --- Topology -----------------------------------------------------------
    topo_counter: Counter[str] = Counter()
    topos_per_row: list[set[str]] = []
    for cell in df[topology_col]:
        codes = _split_topology_cell(cell)
        topos_per_row.append(set(codes))
        topo_counter.update(codes)

    selected_topos, topo_item_cov = _select_by_coverage(
        topo_counter, n_rows, topo_coverage
    )
    topo_full_row_cov = _full_row_coverage(topos_per_row, set(selected_topos))
    topo_any_row_cov = sum(
        1 for s in topos_per_row if s & set(selected_topos)
    ) / n_rows

    # --- Metals -------------------------------------------------------------
    metal_element_counter: Counter[str] = Counter()
    node_smiles_counter: Counter[str] = Counter()
    metals_per_row: list[set[str]] = []
    for cell in df[nodes_col]:
        node_smiles = _parse_list_cell(cell)
        row_metals: set[str] = set()
        for s in node_smiles:
            node_smiles_counter[s] += 1
            for el in _extract_elements(s):
                if el in METALS:
                    metal_element_counter[el] += 1
                    row_metals.add(el)
        metals_per_row.append(row_metals)

    selected_metals, metal_item_cov = _select_by_coverage(
        metal_element_counter, n_rows, metal_coverage
    )
    metal_full_row_cov = _full_row_coverage(metals_per_row, set(selected_metals))
    metal_any_row_cov = sum(
        1 for s in metals_per_row if s & set(selected_metals)
    ) / n_rows

    # --- Linkers ------------------------------------------------------------
    linker_counter: Counter[str] = Counter()
    linkers_per_row: list[set[str]] = []
    raw_linker_total = 0
    canon_failed = 0
    for cell in df[linkers_col]:
        raws = _parse_list_cell(cell)
        row_linkers: set[str] = set()
        for s in raws:
            raw_linker_total += 1
            canon = _canonicalize_smiles(s)
            if canon is None:
                canon_failed += 1
                continue
            linker_counter[canon] += 1
            row_linkers.add(canon)
        linkers_per_row.append(row_linkers)

    top_linkers = [s for s, _ in linker_counter.most_common(linker_top_n)]
    linker_full_row_cov = _full_row_coverage(linkers_per_row, set(top_linkers))
    linker_any_row_cov = sum(
        1 for s in linkers_per_row if s & set(top_linkers)
    ) / n_rows

    # --- PLD / LCD (optional; may be absent for non-QMOF datasets) ----------
    pld = (
        pd.to_numeric(df[pld_col], errors="coerce").dropna()
        if has_pld else pd.Series([], dtype=float)
    )
    lcd = (
        pd.to_numeric(df[lcd_col], errors="coerce").dropna()
        if has_lcd else pd.Series([], dtype=float)
    )
    pore_summary = {
        "pld": {
            "count": int(pld.size),
            "min": float(pld.min()) if pld.size else None,
            "p10": float(pld.quantile(0.10)) if pld.size else None,
            "p50": float(pld.quantile(0.50)) if pld.size else None,
            "p90": float(pld.quantile(0.90)) if pld.size else None,
            "max": float(pld.max()) if pld.size else None,
        },
        "lcd": {
            "count": int(lcd.size),
            "min": float(lcd.min()) if lcd.size else None,
            "p10": float(lcd.quantile(0.10)) if lcd.size else None,
            "p50": float(lcd.quantile(0.50)) if lcd.size else None,
            "p90": float(lcd.quantile(0.90)) if lcd.size else None,
            "max": float(lcd.max()) if lcd.size else None,
        },
    }

    # --- Emit CSVs ----------------------------------------------------------
    pd.DataFrame(topo_counter.most_common(), columns=["topology", "count"]).to_csv(
        out_dir / "topology_counts.csv", index=False
    )
    pd.DataFrame(
        metal_element_counter.most_common(), columns=["metal", "count"]
    ).to_csv(out_dir / "metal_counts.csv", index=False)
    pd.DataFrame(
        node_smiles_counter.most_common(), columns=["node_smiles", "count"]
    ).to_csv(out_dir / "node_smiles_counts.csv", index=False)
    pd.DataFrame(
        linker_counter.most_common(), columns=["linker_smiles", "count"]
    ).to_csv(out_dir / "linker_counts.csv", index=False)
    (out_dir / "pld_lcd_summary.json").write_text(
        json.dumps(pore_summary, indent=2), encoding="utf-8"
    )

    # --- Whitelists ---------------------------------------------------------
    (out_dir / "selected_topologies.txt").write_text(
        "\n".join(selected_topos) + "\n", encoding="utf-8"
    )
    (out_dir / "selected_metals.txt").write_text(
        "\n".join(selected_metals) + "\n", encoding="utf-8"
    )
    (out_dir / "selected_linkers.txt").write_text(
        "\n".join(top_linkers) + "\n", encoding="utf-8"
    )

    # --- Coverage report ----------------------------------------------------
    rows_with_topo = sum(1 for s in topos_per_row if s)
    rows_with_metal = sum(1 for s in metals_per_row if s)
    rows_with_linker = sum(1 for s in linkers_per_row if s)

    report = [
        "# Reference-dataset coverage report",
        "",
        f"- Total rows: **{n_rows}**",
        f"- Rows with topology label: {rows_with_topo} ({rows_with_topo / n_rows:.1%})",
        f"- Rows with a metal node: {rows_with_metal} ({rows_with_metal / n_rows:.1%})",
        f"- Rows with a linker SMILES: {rows_with_linker} ({rows_with_linker / n_rows:.1%})",
        f"- Linker SMILES that RDKit could not canonicalize: {canon_failed}/{raw_linker_total}",
        "",
        "## Topology whitelist",
        f"- Target item-count coverage: {topo_coverage:.0%}",
        f"- Achieved item-count coverage: {topo_item_cov:.1%}",
        f"- Selected {len(selected_topos)} topologies",
        f"- Rows whose topology set intersects the whitelist: {topo_any_row_cov:.1%}",
        f"- Rows whose topology set is fully contained: {topo_full_row_cov:.1%}",
        "",
        "## Metal whitelist",
        f"- Target item-count coverage: {metal_coverage:.0%}",
        f"- Achieved item-count coverage: {metal_item_cov:.1%}",
        f"- Selected {len(selected_metals)} metals",
        f"- Rows whose metal set intersects the whitelist: {metal_any_row_cov:.1%}",
        f"- Rows whose metal set is fully contained: {metal_full_row_cov:.1%}",
        "",
        "## Linker whitelist",
        f"- Top-N cap: {linker_top_n}",
        f"- Selected {len(top_linkers)} canonical linker SMILES "
        f"(out of {len(linker_counter)} unique)",
        f"- Rows whose linker set intersects the whitelist: {linker_any_row_cov:.1%}",
        f"- Rows whose linker set is fully contained: {linker_full_row_cov:.1%}",
        "",
        "## Pore size reference (for downstream screening)",
        "```json",
        json.dumps(pore_summary, indent=2),
        "```",
    ]
    (out_dir / "coverage_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(f"[analyze] wrote whitelists and reports to {out_dir}")
    print(
        f"[analyze] topologies={len(selected_topos)}  "
        f"metals={len(selected_metals)}  linkers={len(top_linkers)}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="Reference MOF table (default: ./qmof.csv).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--topo-coverage", type=float, default=0.95)
    p.add_argument("--metal-coverage", type=float, default=0.99)
    p.add_argument("--linker-top-n", type=int, default=200)
    # Column mapping -- defaults are QMOF's MOFid columns; override to point at
    # any other reference dataset.
    p.add_argument("--topology-col", default="info.mofid.topology",
                   help="Column of comma-separated net/topology codes.")
    p.add_argument("--nodes-col", default="info.mofid.smiles_nodes",
                   help="Column of node SMILES (list literal, bracketed metals).")
    p.add_argument("--linkers-col", default="info.mofid.smiles_linkers",
                   help="Column of organic linker SMILES (list literal).")
    p.add_argument("--pld-col", default="info.pld",
                   help="Optional pore-limiting-diameter column.")
    p.add_argument("--lcd-col", default="info.lcd",
                   help="Optional largest-cavity-diameter column.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(
        args.csv,
        args.out,
        topo_coverage=args.topo_coverage,
        metal_coverage=args.metal_coverage,
        linker_top_n=args.linker_top_n,
        topology_col=args.topology_col,
        nodes_col=args.nodes_col,
        linkers_col=args.linkers_col,
        pld_col=args.pld_col,
        lcd_col=args.lcd_col,
    )
