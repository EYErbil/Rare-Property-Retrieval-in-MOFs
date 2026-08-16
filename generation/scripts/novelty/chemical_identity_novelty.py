"""
chemical_identity_novelty.py -- chemistry-level novelty of the generated low-gap
MOF hits against reference databases, on the joint identity of
NET TOPOLOGY + FRAMEWORK METAL + NEUTRAL-PARENT LINKER.

Complements novelty_check_multidb.py rather than replacing it:

  novelty_check_multidb.py  asks "is this the SAME MATERIAL as a reference?"
      -> reduced-formula prefilter + pymatgen StructureMatcher.
      -> spans every reference structure, at coarse (composition) resolution.

  this script              asks "is this COMBINATION OF BUILDING BLOCKS known?"
      -> net symbol + node metal + neutral-parent linker identity, from MOFid.
      -> finer chemical resolution, but only over annotated references.

Reduced-composition identity is a coarse novelty criterion: a framework
differing only in solvent content, framework stoichiometry, or formula units
per cell has a different reduced composition while being the same chemistry.
This screen therefore compares what actually defines a MOF -- which linker sits
on which metal node in which net.

Method
------
1. Parse each reference entry's MOFid annotation into framework metals,
   linker SMILES, and net symbol. QMOF's explicit ``smiles_nodes`` and
   ``smiles_linkers`` columns carry the same molecular-graph information.
2. Neutralise every query/reference linker with Open Babel, then retain the
   14-character connectivity block of its neutral-parent InChIKey.
3. Describe each generated hit the same way, from its PORMAKE net and node
   metal plus the linker InChIKeys resolved by resolve_linker_smiles.py.
4. A reference counts as a match only if it shares the net symbol AND contains
   the hit's node metal AND contains the hit's linker skeleton.

Neutral-parent normalisation
----------------------------
The first InChIKey block alone is *not* guaranteed to be invariant to
protonation. Before any first-block comparison, both query and reference
linkers are therefore converted from SMILES with ``obabel --neutralize``.
This is essential for N-deprotonated heterocycles: the framework pyrazolate
used by E146 and neutral 1H-pyrazole-3,5-dicarbonitrile otherwise have different
first blocks. Stereochemical layers are intentionally ignored by retaining
only the 14-character connectivity block after neutralisation.

Annotation route
----------------
All databases are read through their MOFid linker SMILES (or QMOF's explicit
``smiles_linkers`` column), because a precomputed MOFkey cannot be reliably
neutralised without the molecular graph. Metals and topology are retained from
the same annotation. Entries without parseable linker SMILES are outside this
chemistry-level screen and are counted as uncovered.

Guardrails -- what this screen does NOT establish
-------------------------------------------------
  * Coverage is not identical to the composition screen. Entries whose upstream
    MOFid perception failed carry no annotation and are absent here. The CoRE
    MOF 2019 distribution ships no MOFid annotations at all; CoRE MOF 2024 ASR
    is a distinct later release sharing only ~33% of CoRE MOF 2019's CSD
    refcodes, so it supplements rather than replaces them.
  * Reference topologies are MOFid-*perceived*; candidate topologies are
    PORMAKE *design* nets. A perception failure could in principle suppress a
    match. Run --validate: the positive controls bound this, by confirming the
    screen recovers established net/metal/linker chemistries.
  * Novelty is scoped to the databases actually searched. No claim is made
    about the CSD or the wider literature.

Run --validate before trusting any null result. A screen that returns zero for
everything is worthless; the controls demonstrate it returns non-zero when a
combination genuinely exists.

Usage
-----
    python scripts/novelty/chemical_identity_novelty.py \
        --qmof-csv      /path/to/qmof.csv \
        --mofdb-json    hmof=/path/to/Dataset/hmof \
        --mofdb-json    tobacco=/path/to/Dataset/tobocco \
        --core2024-csv  /path/to/CoreMOF/mofid-v2/errors/ASR_mofid.csv \
        --validate

Every source is optional; the screen runs over whatever is supplied and always
reports its own coverage. Requires Open Babel (`obabel`) on PATH.
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 7)

SKEL = re.compile(r"^[A-Z]{14}$")
BAD_TOPO = {"", "UNKNOWN", "ERROR", "NA", "NONE"}
METALS = set("""
Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag
Cd In Sn Sb Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au
Hg Tl Pb Bi Th U Np Pu Ac Pa
""".split())

# Query linkers are stored as neutral-parent SMILES and converted through the
# same Open Babel neutralisation path as every reference linker. The expected
# skeletons are regression guards, not values used for matching.
HIT_QUERIES = [
    ("hex+N199+E185", "hex", "Cu",
     "O=C(O)c1c(Br)c(Br)c(C(=O)O)c(Br)c1Br", "PNXPXUDJXYVOFM",
     "tetrabromoterephthalic acid"),
    ("pcu+N273+E44", "pcu", "Mn",
     "O=C(O)C#CC(=O)O", "YTIVTFGABIZHHX", "but-2-ynedioic acid"),
    ("hex+N67+E151", "hex", "Cd",
     "OC(=O)[C@@H]1CC[C@H](CC1)C(=O)O", "PXGZQGDTEZPERC",
     "trans-1,4-cyclohexanedicarboxylic acid"),
    ("pcu+N273+E128", "pcu", "Mn",
     "Nc1cc(C(=O)O)cc(C(=O)O)c1", "KBZFDRWPMZESDI", "5-aminoisophthalic acid"),
    ("cds+N29+E128", "cds", "Mn",
     "Nc1cc(C(=O)O)cc(C(=O)O)c1", "KBZFDRWPMZESDI", "5-aminoisophthalic acid"),
    ("qtz+N307+E146", "qtz", "Cu",
     "N#Cc1[nH]nc(c1)C#N", "LBSASQXIHJDQCN", "1H-pyrazole-3,5-dicarbonitrile"),
]

E146_CHARGED_SMILES = "N#Cc1[n-]nc(c1)C#N"
E146_NEUTRAL_SMILES = "N#Cc1[nH]nc(c1)C#N"

# Established frameworks that MUST be found if the screen is sensitive.
CONTROL_QUERIES = [
    ("MOF-5 / IRMOF-1", "Zn", "O=C(O)c1ccc(cc1)C(=O)O", "KKEYFWRCBNTPAC", "pcu"),
    ("HKUST-1", "Cu", "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1", "QMKYBPDZANOJGF", "tbo"),
    ("UiO-66", "Zr", "O=C(O)c1ccc(cc1)C(=O)O", "KKEYFWRCBNTPAC", "fcu"),
    ("MIL-53(Al)", "Al", "O=C(O)c1ccc(cc1)C(=O)O", "KKEYFWRCBNTPAC", "rna"),
]


# --------------------------------------------------------------------------- #
# annotation parsing
# --------------------------------------------------------------------------- #
def clean_topo(t: str | None) -> str:
    """Topology perception can fail; those entries are not topology-resolvable."""
    t = (t or "").strip()
    return "" if t.upper() in BAD_TOPO else t


def parse_mofkey(mk: str | None):
    """-> (metals, linker skeletons, topology) or None."""
    if not mk or mk in ("None", "null", "NA", "ERROR"):
        return None
    toks = mk.split(".")
    vi = next((i for i, x in enumerate(toks) if x.startswith("MOFkey-v")), None)
    if vi is None:
        return None
    topo = clean_topo(toks[vi + 1] if len(toks) > vi + 1 else "")
    metals, linkers = set(), set()
    for x in toks[:vi]:
        (linkers if SKEL.match(x) else metals).add(x)
    return metals, linkers, topo


def parse_mofid(mofid: str):
    """MOFid string -> (metals, linker SMILES list, topology) or None."""
    if not mofid or " MOFid-v1." not in mofid:
        return None
    left, right = mofid.split(" MOFid-v1.", 1)
    topo = clean_topo(right.split(".")[0].split(";")[0])
    metals, linkers = set(), []
    for comp in (c for c in left.split(".") if c):
        els = set(re.findall(r"\[([A-Z][a-z]?)", comp))
        if els & METALS:
            metals |= els & METALS
        elif "*" not in comp:          # InChI cannot represent wildcard atoms
            linkers.append(comp)
    return metals, linkers, topo


# --------------------------------------------------------------------------- #
# Open Babel
# --------------------------------------------------------------------------- #
def _obabel_batch(batch):
    p = subprocess.run(["obabel", "-ismi", "-oinchikey", "--neutralize"],
                       input="\n".join(batch),
                       capture_output=True, text=True, timeout=3600)
    keys = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    if len(keys) == len(batch):
        return {s: k.split("-")[0] for s, k in zip(batch, keys) if "-" in k}
    # Open Babel voids an ENTIRE batch when one input fails, so fall back per item.
    out = {}
    for s in batch:
        q = subprocess.run(["obabel", "-ismi", "-oinchikey", "--neutralize"], input=s,
                           capture_output=True, text=True, timeout=60)
        k = q.stdout.strip().split("\n")[0].strip()
        if k and "-" in k:
            out[s] = k.split("-")[0]
    return out


def obabel_skeletons(smiles, chunk=200):
    """SMILES -> neutral-parent 14-character InChIKey skeleton, batched."""
    uniq = sorted({s for s in smiles if s and "*" not in s})
    out = {}
    for i in range(0, len(uniq), chunk):
        out.update(_obabel_batch(uniq[i:i + chunk]))
        sys.stderr.write(f"  obabel {min(i + chunk, len(uniq))}/{len(uniq)}\r")
    if uniq:
        sys.stderr.write("\n")
    if len(out) < len(uniq):
        sys.stderr.write(f"  note: {len(uniq) - len(out)} of {len(uniq)} SMILES unconvertible\n")
    return out


def normalized_queries():
    """Normalize hit/control linkers and enforce charged/neutral E146 identity."""
    query_smiles = [q[3] for q in HIT_QUERIES]
    query_smiles.extend(q[2] for q in CONTROL_QUERIES)
    query_smiles.extend((E146_CHARGED_SMILES, E146_NEUTRAL_SMILES))
    keys = obabel_skeletons(query_smiles)

    neutral_key = keys.get(E146_NEUTRAL_SMILES)
    charged_key = keys.get(E146_CHARGED_SMILES)
    if neutral_key != "LBSASQXIHJDQCN" or charged_key != neutral_key:
        raise RuntimeError(
            "E146 neutral-parent regression failed: charged and neutral "
            f"forms resolved to {charged_key!r} and {neutral_key!r}"
        )

    hits = []
    for label, topo, metal, smiles, expected, name in HIT_QUERIES:
        key = keys.get(smiles)
        if key != expected:
            raise RuntimeError(f"{label} linker resolved to {key!r}, expected {expected}")
        hits.append((label, topo, metal, key, name))

    controls = []
    for name, metal, smiles, expected, topo in CONTROL_QUERIES:
        key = keys.get(smiles)
        if key != expected:
            raise RuntimeError(f"{name} linker resolved to {key!r}, expected {expected}")
        controls.append((name, metal, key, topo))
    return hits, controls


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_qmof(path, rows):
    """QMOF explicit linker/node SMILES, neutralized before key comparison."""
    pending = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                lk = ast.literal_eval(r.get("info.mofid.smiles_linkers") or "[]")
                nd = ast.literal_eval(r.get("info.mofid.smiles_nodes") or "[]")
            except (SyntaxError, ValueError):
                lk, nd = [], []
            if not isinstance(lk, (list, tuple)):
                lk = []
            if not isinstance(nd, (list, tuple)):
                nd = []
            if not lk:
                continue
            metals = {
                e for s in nd for e in re.findall(r"\[([A-Z][a-z]?)", s)
                if e in METALS
            }
            mk = parse_mofkey((r.get("info.mofid.mofkey") or "").strip())
            if not metals and mk:
                metals = mk[0]
            topo = clean_topo(r.get("info.mofid.topology")) or (mk[2] if mk else "")
            pending.append((list(lk), metals, topo))
    if pending:
        smi2skel = obabel_skeletons([s for lk, _, _ in pending for s in lk])
        for lk, metals, topo in pending:
            skels = {smi2skel[s] for s in lk if s in smi2skel}
            if skels:
                rows.append(("QMOF", metals, skels, topo))


def load_mofdb_json(name, directory, rows):
    """MOFdb-style JSON exports, using MOFid SMILES rather than raw MOFkeys."""
    parsed, alllk = [], set()
    for fp in glob.glob(os.path.join(directory, "*.json")):
        try:
            with open(fp) as fh:
                d = json.load(fh)
        except Exception:
            continue
        p = parse_mofid(d.get("mofid", ""))
        if p:
            parsed.append(p)
            alllk.update(p[1])
    smi2skel = obabel_skeletons(sorted(alllk))
    n = 0
    for metals, lk, topo in parsed:
        skels = {smi2skel[s] for s in lk if s in smi2skel}
        if skels:
            rows.append((name, metals, skels, topo))
            n += 1
    sys.stderr.write(f"  {name}: {n}\n")


def load_core2024(path, rows):
    """CoRE MOF 2024 ASR published MOFid strings (cifname,mofid)."""
    parsed, alllk = [], set()
    with open(path) as f:
        for r in csv.DictReader(f):
            p = parse_mofid(r.get("mofid", ""))
            if p:
                parsed.append(p)
                alllk.update(p[1])
    smi2skel = obabel_skeletons(sorted(alllk))
    n = 0
    for metals, lk, topo in parsed:
        skels = {smi2skel[s] for s in lk if s in smi2skel}
        if skels:
            rows.append(("CoRE2024", metals, skels, topo))
            n += 1
    sys.stderr.write(f"  CoRE2024: {n}\n")


# --------------------------------------------------------------------------- #
def tally(rows, metal, linker, topo):
    c = defaultdict(int)
    for _db, M, L, T in rows:
        m, l = metal in M, linker in L
        c["metal"] += m
        c["linker"] += l
        c["metal+linker"] += (m and l)
        if T:
            t = (T == topo)
            c["net"] += t
            c["net+linker"] += (t and l)
            c["net+metal"] += (t and m)
            c["net+metal+linker"] += (t and m and l)
    return c


HDR = (f"  {'name':17s} {'net':4s} {'M':3s} {'linker key':15s} "
       f"{'metal':>8s} {'linker':>7s} {'net':>7s} {'m+l':>6s} {'n+l':>6s} {'n+m':>7s} {'n+m+l':>6s}")


def line(name, topo, metal, lk, c, suffix=""):
    return (f"  {name:17s} {topo:4s} {metal:3s} {lk:15s} "
            f"{c['metal']:>8,d} {c['linker']:>7,d} {c['net']:>7,d} "
            f"{c['metal+linker']:>6,d} {c['net+linker']:>6,d} {c['net+metal']:>7,d} "
            f"{c['net+metal+linker']:>6,d}{suffix}")


def main():
    ap = argparse.ArgumentParser(
        description="Net+metal+linker chemical-identity novelty screen for the "
                    "generated low-gap MOF hits.")
    ap.add_argument("--qmof-csv", type=Path,
                    help="QMOF csv with info.mofid.* columns")
    ap.add_argument("--mofdb-json", type=str, action="append", default=[],
                    metavar="NAME=DIR",
                    help="MOFdb JSON export as name=dir; repeat per database "
                         "(e.g. hmof=..., tobacco=...)")
    ap.add_argument("--core2024-csv", type=Path,
                    help="CoRE MOF 2024 ASR MOFid csv (cifname,mofid)")
    ap.add_argument("--validate", action="store_true",
                    help="run positive controls; do this before trusting a null result")
    ap.add_argument("--out", type=Path, help="write the report here as well as stdout")
    args = ap.parse_args()

    if not (args.qmof_csv or args.mofdb_json or args.core2024_csv):
        ap.error("supply at least one reference source")

    sys.stderr.write("normalizing candidate and control linker queries...\n")
    hits, controls = normalized_queries()

    rows = []
    sys.stderr.write("loading reference annotations...\n")
    if args.qmof_csv:
        load_qmof(args.qmof_csv, rows)
    for spec in args.mofdb_json:
        if "=" not in spec:
            ap.error(f"--mofdb-json expects NAME=DIR, got {spec!r}")
        name, d = spec.split("=", 1)
        load_mofdb_json(name, d, rows)
    if args.core2024_csv:
        load_core2024(args.core2024_csv, rows)
    if not rows:
        sys.exit("no annotated reference entries loaded")

    buf = []
    def emit(s=""):
        buf.append(s)
        print(s)

    by_db, by_topo = defaultdict(int), defaultdict(int)
    for r in rows:
        by_db[r[0]] += 1
        if r[3]:
            by_topo[r[0]] += 1
    n_topo = sum(by_topo.values())

    emit("=" * 100)
    emit("REFERENCE COVERAGE")
    emit("=" * 100)
    emit(f"  {'database':10s} {'metal+linker':>13s} {'+topology':>11s}")
    for k in sorted(by_db):
        emit(f"  {k:10s} {by_db[k]:>13,d} {by_topo[k]:>11,d}")
    emit(f"  {'TOTAL':10s} {len(rows):>13,d} {n_topo:>11,d}")
    emit("  Coverage is limited to entries carrying a parseable annotation;")
    emit("  see the module docstring for what this screen does not establish.")

    if args.validate:
        emit()
        emit("=" * 100)
        emit("POSITIVE CONTROLS (established frameworks; the screen must find these)")
        emit("=" * 100)
        emit(HDR)
        failed = []
        for name, metal, lk, topo in controls:
            c = tally(rows, metal, lk, topo)
            ok = c["net+metal+linker"] > 0
            if not ok:
                failed.append(name)
            emit(line(name, topo, metal, lk, c, "   PASS" if ok else "   **FAIL**"))
        if failed:
            emit(f"  WARNING: controls failed ({', '.join(failed)}); "
                 f"null results below are NOT interpretable as absence.")

    emit()
    emit("=" * 100)
    emit("HITS: references matching on metal, linker, net and their combinations")
    emit("=" * 100)
    emit(HDR)
    for label, topo, metal, lk, _ln in hits:
        emit(line(label, topo, metal, lk, tally(rows, metal, lk, topo)))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(buf) + "\n")
        sys.stderr.write(f"wrote {args.out}\n")


if __name__ == "__main__":
    main()
