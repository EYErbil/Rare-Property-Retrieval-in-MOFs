"""
novelty_check_multidb.py -- structural novelty of the generated low-gap MOF hits
against MULTIPLE reference MOF databases (QMOF, CoRE MOF 2019, hMOF, ToBaCCo).

Keeps PER-DATABASE attribution (rather than merging every reference into one
bucket), so the manuscript can state, for each database searched, whether a
generated hit was already present (and, if so, which entry).

Method (mirrors the Methods section of the paper):
  For each generated framework G and each reference database D:
    1. reduced-formula prefilter -- collect the refs in D that share G's
       pymatgen reduced formula ("composition collisions"). This avoids the
       O(N_gen x N_ref) cost of full structure matching.
    2. pymatgen StructureMatcher (MOF-tolerant: ltol, stol, angle_tol;
       primitive_cell + attempt_supercell) against ONLY those collisions.
    3. record: #collisions, whether any exact structure match, matched ref id.
  A hit is "novel vs the searched databases" iff it matches no structure in
  ANY searched D. We never claim novelty beyond the databases actually
  searched -- CSD and the full literature are larger (see README guardrail).

Why this is fast even on hMOF (~10^5 cifs):
  * Each database is indexed ONCE into {reduced_formula -> [paths]} and the
    index is cached to JSON PER DATABASE. Re-runs, added query structures, or
    a crash mid-way are cheap -- a finished database is never re-walked.
  * The index build reads each reference's reduced formula via a cheap text
    scan of the CIF '_chemical_formula_sum' tag, and only falls back to a full
    pymatgen parse when that tag is missing. Full structure parsing is reserved
    for the few composition collisions that actually need StructureMatcher.

  *** Do NOT run until the Dataset/ databases finish extracting. ***
  This script only PREPARES the analysis. Missing/empty reference dirs are
  skipped with a warning, so a partial run while extraction is ongoing is safe
  but incomplete. Use --dry-run to sanity-check paths and query count without
  indexing anything.

Run (venv with pymatgen):
    python scripts/novelty/novelty_check_multidb.py \
        --query-dir /path/to/candidate_cifs \
        --ref-db qmof=/path/to/qmof_cifs \
        --ref-db coremof2019=/path/to/CoreMof2019 \
        --ref-db hmof=/path/to/hmof \
        --ref-db tobacco=/path/to/tobacco

Dependencies: pymatgen, pandas.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # scripts/
REPO = ROOT.parent                      # repository root

# ----- defaults -------------------------------------------------------------
# Query = the CIFs whose novelty is being tested. For the paper these were the
# five HSE06-confirmed generated hits (distributed with the results archive;
# see Data availability). The QMOF-pool hits are already-synthesized QMOF
# entries, so a novelty test against external databases is not meaningful for
# them. Point --query-dir at any directory of candidate structures.
DEFAULT_QUERY_DIR = (REPO / "candidate_cifs").resolve()

# Reference databases are large and NOT distributed with this repository —
# download QMOF, CoRE MOF 2019, hMOF, and ToBaCCo yourself and point --ref-db
# at your local CIF directories (any subset works; all four for the paper).
DEFAULT_REF_DBS = {
    "coremof2019": (REPO / "Dataset" / "CoreMof2019"),
    "hmof":        (REPO / "Dataset" / "hmof"),
    "tobacco":     (REPO / "Dataset" / "tobacco"),
}

CACHE_DIR  = (REPO / "data" / "processed" / "novelty_index").resolve()
DEFAULT_OUT = (REPO / "novelty_results" / "novelty_multidb.md").resolve()
DEFAULT_CSV = (REPO / "novelty_results" / "novelty_multidb.csv").resolve()
DEFAULT_LOG = (REPO / "logs" / "novelty_multidb.log").resolve()

STRUCT_PATTERNS = ("*.cif", "POSCAR", "CONTCAR", "*.vasp")

# CIF tag carrying the unit-cell summed formula, e.g. "_chemical_formula_sum 'C8 H4 Cu2 O8'"
_FORMULA_SUM_RE = re.compile(
    r"_chemical_formula_sum\s+['\"]?\s*([A-Za-z0-9 .()]+?)\s*['\"]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# ----- logging --------------------------------------------------------------
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("novelty_multidb")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s")
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


# ----- structure discovery / parsing ---------------------------------------
def find_structures(d: Path) -> List[Path]:
    """All structure files under d (recursively), de-duplicated and sorted."""
    if not d.exists():
        return []
    out: List[Path] = []
    for pat in STRUCT_PATTERNS:
        out.extend(d.rglob(pat))
    seen, uniq = set(), []
    for p in sorted(out):
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def label_for(path: Path) -> str:
    if path.name in {"POSCAR", "CONTCAR"}:
        return path.parent.name
    return path.stem


def full_structure(path: Path):
    """Full pymatgen Structure (needed for StructureMatcher). None on failure."""
    from pymatgen.core import Structure
    try:
        return Structure.from_file(str(path))
    except Exception:
        return None


def quick_reduced_formula(path: Path) -> Optional[str]:
    """
    Cheap reduced formula for indexing references.

    Tries the CIF '_chemical_formula_sum' tag first (a text scan, no structure
    build); falls back to a full pymatgen parse only when the tag is absent or
    unparseable. POSCAR/CONTCAR/.vasp always take the full-parse path.
    Returns None if the structure cannot be read at all.
    """
    try:
        if path.suffix.lower() == ".cif":
            text = path.read_text(errors="ignore")
            m = _FORMULA_SUM_RE.search(text)
            if m:
                raw = m.group(1).strip()
                if raw and raw not in {"?", "."}:
                    from pymatgen.core import Composition
                    try:
                        return Composition(raw).reduced_formula
                    except Exception:
                        pass  # fall through to full parse
    except Exception:
        pass
    s = full_structure(path)
    return s.composition.reduced_formula if s is not None else None


def _formula_worker(path_str: str) -> Tuple[str, Optional[str]]:
    return path_str, quick_reduced_formula(Path(path_str))


# ----- per-database reduced-formula index (cached) --------------------------
def build_index(name: str, ref_dir: Path, cache_dir: Path, jobs: int,
                logger: logging.Logger) -> Optional[Dict[str, List[str]]]:
    """
    {reduced_formula -> [abs paths]} for one database, cached to JSON.

    Returns None if the directory is missing/empty (e.g. still extracting),
    so the caller can skip it and continue with the other databases.
    """
    cache = cache_dir / f"index_{name}.json"
    if cache.exists():
        logger.info("[%s] loading cached index: %s", name, cache)
        return json.loads(cache.read_text())

    paths = find_structures(ref_dir)
    if not paths:
        logger.warning("[%s] no structures under %s -- skipping "
                       "(still extracting?)", name, ref_dir)
        return None

    logger.info("[%s] indexing %d reference structures from %s (jobs=%d)...",
                name, len(paths), ref_dir, jobs)
    index: Dict[str, List[str]] = defaultdict(list)
    n_ok = n_bad = 0

    def _record(path_str: str, rf: Optional[str]) -> None:
        nonlocal n_ok, n_bad
        if rf is None:
            n_bad += 1
        else:
            index[rf].append(path_str); n_ok += 1
        done = n_ok + n_bad
        if done % 2000 == 0:
            logger.info("  [%s] %d/%d indexed (%d unreadable)...",
                        name, done, len(paths), n_bad)

    path_strs = [str(p) for p in paths]
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for fut in as_completed(ex.submit(_formula_worker, ps)
                                    for ps in path_strs):
                ps, rf = fut.result()
                _record(ps, rf)
    else:
        for ps in path_strs:
            _record(ps, quick_reduced_formula(Path(ps)))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index))
    logger.info("[%s] indexed %d structures into %d formula buckets "
                "(%d unreadable) -> %s",
                name, n_ok, len(index), n_bad, cache)
    return index


# ----- argument helpers -----------------------------------------------------
def parse_ref_db(spec: str) -> Tuple[str, Path]:
    """'name=dir' -> (name, resolved Path). Bare 'dir' -> (dir.name, Path)."""
    if "=" in spec:
        name, d = spec.split("=", 1)
    else:
        d = spec; name = Path(spec).name
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_").lower()
    return name, Path(d).expanduser().resolve()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-database StructureMatcher novelty check for the "
                    "generated low-gap MOF hits.")
    ap.add_argument("--query-dir", type=Path, action="append", default=[],
                    help="dir(s) of generated structures to test "
                         f"(default: {DEFAULT_QUERY_DIR})")
    ap.add_argument("--ref-db", type=str, action="append", default=[],
                    metavar="NAME=DIR",
                    help="reference database as name=dir; repeat per database. "
                         "If omitted, uses the Dataset/ defaults "
                         "(coremof2019, hmof, tobacco).")
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle-tol", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel workers for index building (default 4)")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--dry-run", action="store_true",
                    help="report query count and which reference dirs are "
                         "present/non-empty, then exit (no indexing/matching).")
    args = ap.parse_args()

    logger = setup_logger(args.log)

    # resolve query dirs
    query_dirs = args.query_dir or [DEFAULT_QUERY_DIR]
    query_paths: List[Path] = []
    for qd in query_dirs:
        found = find_structures(Path(qd).expanduser().resolve())
        logger.info("query dir %s -> %d structures", qd, len(found))
        query_paths.extend(found)
    if not query_paths:
        logger.error("no query structures found under %s", query_dirs)
        return 2

    # resolve reference databases
    if args.ref_db:
        ref_dbs = [parse_ref_db(s) for s in args.ref_db]
    else:
        ref_dbs = [(n, p.resolve()) for n, p in DEFAULT_REF_DBS.items()]
    logger.info("reference databases: %s",
                ", ".join(f"{n}({d})" for n, d in ref_dbs))

    # ---- dry run: just check what is present, do not index ----
    if args.dry_run:
        logger.info("DRY RUN -- query structures: %d", len(query_paths))
        for ps in query_paths:
            logger.info("    query: %s", label_for(ps))
        for name, d in ref_dbs:
            n = len(find_structures(d))
            status = "MISSING" if not d.exists() else (
                "EMPTY (extracting?)" if n == 0 else f"{n} structures")
            logger.info("    ref [%-12s] %s -> %s", name, d, status)
        logger.info("DRY RUN complete -- nothing indexed or matched.")
        return 0

    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
    except Exception as e:
        logger.error("pymatgen not available (%s). pip install pymatgen", e)
        return 2

    sm = StructureMatcher(ltol=args.ltol, stol=args.stol,
                          angle_tol=args.angle_tol,
                          primitive_cell=True, attempt_supercell=True)

    # parse the (few) query structures once
    queries: List[Tuple[str, str, object]] = []   # (label, reduced_formula, Structure)
    for qp in query_paths:
        gs = full_structure(qp)
        if gs is None:
            logger.warning("skip unreadable query %s", qp); continue
        queries.append((label_for(qp), gs.composition.reduced_formula, gs))
    logger.info("loaded %d query structures", len(queries))

    # ---- per database: index, prefilter, match ----
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []                                  # one row per (query, db)
    searched_dbs: List[str] = []
    skipped_dbs: List[str] = []
    for name, ref_dir in ref_dbs:
        index = build_index(name, ref_dir, args.cache_dir, args.jobs, logger)
        if index is None:
            skipped_dbs.append(name)
            continue
        searched_dbs.append(name)
        for label, rf, gs in queries:
            candidates = index.get(rf, [])
            match_path: Optional[str] = None
            for cp in candidates:
                rs = full_structure(Path(cp))
                if rs is not None and sm.fit(gs, rs):
                    match_path = cp; break
            rows.append({
                "query": label,
                "database": name,
                "reduced_formula": rf,
                "composition_collisions": len(candidates),
                "match_in_db": "YES" if match_path else "no",
                "matched_reference": Path(match_path).name if match_path else "",
            })
            logger.info("%-20s vs %-12s formula=%-16s collisions=%-5d match=%s",
                        label, name, rf, len(candidates),
                        rows[-1]["match_in_db"])

    if not searched_dbs:
        logger.error("no reference databases were searchable (all missing/"
                     "empty). Are the Dataset/ folders still extracting?")
        return 2

    # ---- write CSV ----
    import pandas as pd
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)

    # ---- per-query verdict across all searched databases ----
    verdict = {}
    for label, rf, _ in queries:
        sub = df[df["query"] == label]
        matched_in = sub.loc[sub["match_in_db"] == "YES", "database"].tolist()
        collided_in = sub.loc[sub["composition_collisions"] > 0,
                              "database"].tolist()
        verdict[label] = {
            "reduced_formula": rf,
            "matched_in": matched_in,
            "collided_in": collided_in,
            "novel": len(matched_in) == 0,
        }
    n_novel = sum(v["novel"] for v in verdict.values())

    # ---- markdown report ----
    L = [f"# Multi-database novelty check (Script 24)\n\n",
         f"StructureMatcher(ltol={args.ltol}, stol={args.stol}, "
         f"angle_tol={args.angle_tol}, primitive_cell=True, "
         f"attempt_supercell=True)\n\n",
         f"Databases searched: {', '.join(searched_dbs)}"
         + (f"  |  SKIPPED (missing/empty): {', '.join(skipped_dbs)}"
            if skipped_dbs else "") + "\n\n",
         f"**{n_novel}/{len(queries)} generated hits matched no structure in "
         f"any searched database.**\n\n"]

    # matrix: query x database
    L.append("## Match matrix (per database)\n\n")
    header = "| query | reduced_formula | " + \
             " | ".join(searched_dbs) + " |\n"
    L.append(header)
    L.append("|" + "---|" * (2 + len(searched_dbs)) + "\n")
    for label, rf, _ in queries:
        cells = []
        for name in searched_dbs:
            r = df[(df["query"] == label) & (df["database"] == name)]
            if r.empty:
                cells.append("-")
            else:
                r = r.iloc[0]
                if r["match_in_db"] == "YES":
                    cells.append(f"**MATCH** ({r['matched_reference']})")
                elif r["composition_collisions"] > 0:
                    cells.append(f"no ({int(r['composition_collisions'])} comp.)")
                else:
                    cells.append("no (0 comp.)")
        L.append(f"| {label} | {rf} | " + " | ".join(cells) + " |\n")

    # per-query plain-English verdict
    L.append("\n## Per-hit verdict\n\n")
    for label, v in verdict.items():
        if v["novel"]:
            extra = (f" and shares a reduced composition only with "
                     f"{v['collided_in']}" if v["collided_in"]
                     else " and shares no reduced composition with any entry")
            L.append(f"- **{label}** ({v['reduced_formula']}): novel vs "
                     f"{searched_dbs}{extra}.\n")
        else:
            L.append(f"- **{label}** ({v['reduced_formula']}): MATCHES an entry "
                     f"in {v['matched_in']}.\n")

    L += ["\n## Allowed phrasing\n",
          f"- 'None of the {len(queries)} generated frameworks matched any "
          f"structure in the searched databases ({', '.join(searched_dbs)}; "
          f"pymatgen StructureMatcher, ltol={args.ltol}, stol={args.stol}, "
          f"angle tol={args.angle_tol} deg).'\n",
          "\n## Disallowed (do NOT claim)\n",
          "- 'definitely novel' / 'never reported' -- the CSD and the full "
          "experimental literature are not searched here. Keep claims scoped "
          "to the databases actually searched.\n"]
    if skipped_dbs:
        L.append(f"\n> NOTE: {skipped_dbs} were missing/empty at run time "
                 "(still extracting?). Re-run after extraction to include them; "
                 "cached databases are not re-indexed.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(L), encoding="utf-8")
    logger.info("wrote %s and %s", args.out, args.csv)
    logger.info("DONE: %d/%d generated hits unmatched across {%s}",
                n_novel, len(queries), ", ".join(searched_dbs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
