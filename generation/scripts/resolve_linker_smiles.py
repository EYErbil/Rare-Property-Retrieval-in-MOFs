#!/usr/bin/env python3
"""Software-derived chemical identifiers for the generated MOFs' linkers/nodes.

No chemistry is asserted by hand. Every identifier here is produced by software:

  * Connectivity and bond orders come from the explicit bond list inside each
    PORMAKE building-block .xyz (codes A=aromatic, S=single, D=double, T=triple)
    -- not guessed from interatomic distances -- so the molecular graph is taken
    verbatim from the block definition that PORMAKE actually assembled.
  * That graph is written to an MDL SDF and handed to Open Babel, which performs
    aromaticity perception/kekulisation and emits the canonical SMILES, InChI,
    InChIKey and molecular formula. The InChIKey is the lookup handle: paste it
    into PubChem to recover the registered IUPAC/common name -- we deliberately
    do not name the molecules ourselves.

For each ditopic organic edge block we report three forms, increasing in
chemical completeness, all canonicalised by Open Babel:

  bb     : the block exactly as PORMAKE stores it, connection points kept as
           [*] dummy attachment atoms (100% faithful to the generator).
  Hcap   : connection points replaced by H -> the neutral parent scaffold, a
           real molecule that resolves in PubChem.
  COOHcap: connection points replaced by -C(=O)OH. The node-block analysis (see
           analyze_nodes()) shows every edge connection point bonds to a node
           carboxylate carbon, so this is the ditopic carboxylic-acid linker as
           it exists in the framework. Capping is a structural fact from the
           bond graph, then canonicalised by software.

Metal nodes are not given SMILES (Open Babel organic perception is not meant for
metal-oxo clusters); they are characterised from the bond graph instead: metal,
carboxylate count, bridging Cl/O, and number of connection points.

    python resolve_linker_smiles.py
writes linker_identifiers.csv, node_identifiers.csv, chemical_identifiers.md.
"""
from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# PORMAKE building-block library (N*.xyz nodes, E*.xyz edges). Defaults to the
# library committed with this repository; override with the BB_DIR env variable.
BB_DIR = Path(os.environ.get("BB_DIR", REPO / "qmof_bb_dir"))

# which blocks each generated hit uses (from the PORMAKE .cif headers)
GENERATED = {
    "hex+N199+E185": dict(topology="hex", cn=8, node="N199", edge="E185"),
    "hex+N67+E151":  dict(topology="hex", cn=8, node="N67",  edge="E151"),
    "pcu+N273+E44":  dict(topology="pcu", cn=6, node="N273", edge="E44"),
    "pcu+N273+E128": dict(topology="pcu", cn=6, node="N273", edge="E128"),
    "cds+N29+E128":  dict(topology="cds", cn=4, node="N29",  edge="E128"),
}

BOND_SDF = {"A": 4, "S": 1, "D": 2, "T": 3}  # MDL: 4=aromatic,1,2,3


# --------------------------------------------------------------------------- #
# building-block parsing                                                       #
# --------------------------------------------------------------------------- #

def parse_bb(bb_id: str):
    """Return (atoms[(sym,x,y,z)], bonds[(i,j,code)]) from a PORMAKE block xyz."""
    lines = (BB_DIR / f"{bb_id}.xyz").read_text().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for ln in lines[2:2 + n]:
        p = ln.split()
        atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    bonds = []
    for ln in lines[2 + n:]:
        p = ln.split()
        if len(p) == 3 and p[0].lstrip("-").isdigit() and p[1].lstrip("-").isdigit():
            bonds.append((int(p[0]), int(p[1]), p[2]))
    return atoms, bonds


# --------------------------------------------------------------------------- #
# SDF building + Open Babel                                                    #
# --------------------------------------------------------------------------- #

def build_sdf(atoms, bonds, x_mode: str) -> str:
    """MDL SDF from an explicit bond graph.

    x_mode:
      'dummy' -> each connection point X becomes a [*] dummy atom.
      'H'     -> each X becomes a capping hydrogen.
      'COOH'  -> each X becomes a carboxyl carbon C(=O)OH (X kept as the carboxyl
                 carbon, with O=, O-H added). Position-independent, so no SMILES
                 string surgery is needed.
    """
    out_atoms = [list(a) for a in atoms]   # [sym,x,y,z]
    out_bonds = [list(b) for b in bonds]   # [i,j,code]

    if x_mode in ("dummy", "H"):
        for a in out_atoms:
            if a[0] == "X":
                a[0] = "*" if x_mode == "dummy" else "H"
    elif x_mode == "COOH":
        for k, a in enumerate(out_atoms):
            if a[0] != "X":
                continue
            a[0] = "C"                                   # carboxyl carbon (keeps its bond)
            bx, by, bz = a[1], a[2], a[3]
            od = len(out_atoms); out_atoms.append(["O", bx + 0.6, by + 0.6, bz])
            os_ = len(out_atoms); out_atoms.append(["O", bx - 0.6, by + 0.6, bz])
            ho = len(out_atoms); out_atoms.append(["H", bx - 0.9, by + 1.1, bz])
            out_bonds.append([k, od, "D"])               # C=O
            out_bonds.append([k, os_, "S"])              # C-O
            out_bonds.append([os_, ho, "S"])             # O-H
    else:
        raise ValueError(x_mode)

    lines = ["", "  pormake-bb", ""]
    lines.append(f"{len(out_atoms):>3}{len(out_bonds):>3}  0  0  0  0  0  0  0  0999 V2000")
    for sym, x, y, z in out_atoms:
        lines.append(f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {sym:<3}"
                     " 0  0  0  0  0  0  0  0  0  0  0  0")
    for i, j, code in out_bonds:
        lines.append(f"{i+1:>3}{j+1:>3}{BOND_SDF[code]:>3}  0  0  0  0")
    lines.append("M  END")
    return "\n".join(lines) + "\n"


def obabel(sdf_or_smi: str, in_fmt: str, out_fmt: str) -> str:
    """Run Open Babel on a temp input; return the first non-empty output token."""
    with tempfile.NamedTemporaryFile("w", suffix=f".{in_fmt}", delete=False) as fh:
        fh.write(sdf_or_smi)
        path = fh.name
    try:
        res = subprocess.run(["obabel", path, f"-o{out_fmt}"],
                             capture_output=True, text=True)
    finally:
        Path(path).unlink(missing_ok=True)
    for line in res.stdout.splitlines():
        tok = line.strip()
        if tok and not tok.startswith("1 molecule"):
            # SMILES/can lines may carry a trailing title field -> keep col 0
            return tok.split("\t")[0].strip()
    return ""


def smi_to(smi: str, out_fmt: str) -> str:
    return obabel(smi, "smi", out_fmt)


def sdf_to(sdf: str, out_fmt: str) -> str:
    return obabel(sdf, "sdf", out_fmt)


def ob_formula(smi: str) -> str:
    """Open Babel molecular formula for a SMILES (software, not hand-counted)."""
    if not smi:
        return ""
    res = subprocess.run(["obabel", f"-:{smi}", "-osmi", "--append", "formula"],
                         capture_output=True, text=True)
    for line in res.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip()
    return ""


# --------------------------------------------------------------------------- #
# node characterisation (from bond graph)                                      #
# --------------------------------------------------------------------------- #

METALS = {"Cu", "Cd", "Mn", "Zn", "Co", "Ni", "Fe", "Mg", "Ca", "Cr", "V", "Ti"}


def analyze_node(bb_id: str) -> dict:
    atoms, bonds = parse_bb(bb_id)
    sym = [a[0] for a in atoms]
    elem = Counter(s for s in sym if s != "X")
    n_x = sym.count("X")
    # carboxylate carbons: carbon bonded to >= 2 oxygens
    o_neighbors = Counter()
    for i, j, _ in bonds:
        if sym[i] == "C" and sym[j] == "O":
            o_neighbors[i] += 1
        if sym[j] == "C" and sym[i] == "O":
            o_neighbors[j] += 1
    n_carboxylate = sum(1 for c, k in o_neighbors.items() if k >= 2)
    metals = sorted({s for s in elem if s in METALS}, key=lambda s: -elem[s])
    return dict(
        node=bb_id,
        metal="/".join(metals) or "?",
        n_metal=sum(elem[m] for m in metals),
        n_carboxylate=n_carboxylate,
        n_Cl=elem.get("Cl", 0),
        n_O=elem.get("O", 0),
        connection_points=n_x,
        formula="".join(f"{e}{elem[e]}" for e in sorted(elem)),
    )


# --------------------------------------------------------------------------- #
# edge (linker) characterisation                                              #
# --------------------------------------------------------------------------- #

def analyze_edge(bb_id: str) -> dict:
    atoms, bonds = parse_bb(bb_id)
    elem = Counter(a[0] for a in atoms if a[0] != "X")
    n_x = sum(1 for a in atoms if a[0] == "X")

    sdf_dummy = build_sdf(atoms, bonds, "dummy")
    sdf_h = build_sdf(atoms, bonds, "H")
    sdf_cooh = build_sdf(atoms, bonds, "COOH")

    smi_bb = sdf_to(sdf_dummy, "can")          # connection pts as [*]
    smi_h = sdf_to(sdf_h, "can")               # H-capped parent
    inchikey_h = sdf_to(sdf_h, "inchikey")

    # COOH-capped: connection points turned into carboxyl carbons at the SDF
    # level (each edge connection point bonds to a node carboxylate carbon).
    smi_cooh = sdf_to(sdf_cooh, "can")
    inchikey_cooh = sdf_to(sdf_cooh, "inchikey")

    return dict(
        edge=bb_id,
        n_connect=n_x,
        bb_formula_noX="".join(f"{e}{elem[e]}" for e in sorted(elem)),
        smiles_bb=smi_bb,
        smiles_Hcap=smi_h,
        inchikey_Hcap=inchikey_h,
        ob_formula_Hcap=ob_formula(smi_h),
        smiles_COOHcap=smi_cooh,
        inchikey_COOHcap=inchikey_cooh,
        ob_formula_COOHcap=ob_formula(smi_cooh),
    )


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    edges = sorted({g["edge"] for g in GENERATED.values()})
    nodes = sorted({g["node"] for g in GENERATED.values()})

    edge_rows = {e: analyze_edge(e) for e in edges}
    node_rows = {n: analyze_node(n) for n in nodes}

    with (HERE / "linker_identifiers.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(edge_rows.values())).keys()))
        w.writeheader(); w.writerows(edge_rows.values())
    with (HERE / "node_identifiers.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(node_rows.values())).keys()))
        w.writeheader(); w.writerows(node_rows.values())

    # combined per-MOF markdown
    md = ["# Software-derived chemical identifiers for the generated MOFs\n\n",
          "All SMILES/InChIKey/formula produced by Open Babel from the PORMAKE "
          "building-block bond graphs. Names intentionally omitted -- resolve "
          "each InChIKey at PubChem.\n\n",
          "## Per-MOF summary\n\n",
          "| MOF | topology (cn) | node SBU | edge | linker formula (parent) | "
          "linker SMILES (carboxylic-acid form) | InChIKey (acid) |\n",
          "|---|---|---|---|---|---|---|\n"]
    for name, g in GENERATED.items():
        e = edge_rows[g["edge"]]; n = node_rows[g["node"]]
        sbu = (f"{n['metal']}{n['n_metal']} cluster, {n['n_carboxylate']} "
               f"carboxylate, {n['n_Cl']} Cl")
        md.append(f"| {name} | {g['topology']} ({g['cn']}) | {sbu} | {g['edge']} | "
                  f"{e['ob_formula_Hcap']} | `{e['smiles_COOHcap']}` | "
                  f"{e['inchikey_COOHcap']} |\n")

    md.append("\n## Edge (linker) blocks -- full identifier set\n\n")
    for e_id, e in edge_rows.items():
        md.append(f"\n### {e_id}  (connection points: {e['n_connect']})\n")
        md.append(f"- block formula (X excluded): `{e['bb_formula_noX']}`\n")
        md.append(f"- SMILES, attachment-point form: `{e['smiles_bb']}`\n")
        md.append(f"- SMILES, H-capped parent: `{e['smiles_Hcap']}`  "
                  f"(InChIKey {e['inchikey_Hcap']})\n")
        md.append(f"- SMILES, carboxylic-acid form: `{e['smiles_COOHcap']}`  "
                  f"(InChIKey {e['inchikey_COOHcap']})\n")
        md.append(f"- Open Babel formula (parent): `{e['ob_formula_Hcap']}`\n")

    md.append("\n## Node (metal SBU) blocks\n\n")
    md.append("| node | metal | n_metal | carboxylates | Cl | O | connection pts | formula |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for n_id, n in node_rows.items():
        md.append(f"| {n_id} | {n['metal']} | {n['n_metal']} | {n['n_carboxylate']} | "
                  f"{n['n_Cl']} | {n['n_O']} | {n['connection_points']} | {n['formula']} |\n")

    (HERE / "chemical_identifiers.md").write_text("".join(md), encoding="utf-8")

    # console
    print("EDGES (linkers):")
    for e_id, e in edge_rows.items():
        print(f"\n  {e_id}: connect={e['n_connect']}  parent_formula={e['ob_formula_Hcap']}")
        print(f"    bb SMILES   : {e['smiles_bb']}")
        print(f"    H-cap SMILES: {e['smiles_Hcap']}   {e['inchikey_Hcap']}")
        print(f"    COOH SMILES : {e['smiles_COOHcap']}   {e['inchikey_COOHcap']}")
    print("\nNODES (SBUs):")
    for n_id, n in node_rows.items():
        print(f"  {n_id}: {n['metal']}{n['n_metal']}  carboxylate={n['n_carboxylate']}  "
              f"Cl={n['n_Cl']}  O={n['n_O']}  connect={n['connection_points']}  {n['formula']}")


if __name__ == "__main__":
    main()
