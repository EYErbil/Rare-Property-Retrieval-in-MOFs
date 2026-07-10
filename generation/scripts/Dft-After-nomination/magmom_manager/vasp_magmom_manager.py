#!/usr/bin/env python3

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_MOMENTS = {
    # Organic / ligand / common closed-shell elements
    "H": 0.0, "B": 0.0, "C": 0.0, "N": 0.0, "O": 0.0, "F": 0.0,
    "Si": 0.0, "P": 0.0, "S": 0.0, "Cl": 0.0, "Br": 0.0, "I": 0.0,

    # Alkali / alkaline earth
    "Li": 0.0, "Na": 0.0, "K": 0.0, "Rb": 0.0, "Cs": 0.0,
    "Be": 0.0, "Mg": 0.0, "Ca": 0.0, "Sr": 0.0, "Ba": 0.0,

    # Main group / closed shell-ish
    "Al": 0.0, "Ga": 0.0, "In": 0.0, "Sn": 0.0, "Pb": 0.0,
    "Zn": 0.0, "Cd": 0.0, "Hg": 0.0,
    "Ag": 0.0, "Au": 0.0,

    # 3d transition metals: starting guesses
    "Sc": 1.0, "Ti": 2.0, "V": 3.0, "Cr": 5.0, "Mn": 5.0,
    "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0,

    # 4d / 5d rough guesses
    "Y": 0.0, "Zr": 1.0, "Nb": 2.0, "Mo": 2.0,
    "Tc": 3.0, "Ru": 2.0, "Rh": 1.0, "Pd": 0.0,
    "Hf": 1.0, "Ta": 2.0, "W": 2.0, "Re": 3.0,
    "Os": 2.0, "Ir": 1.0, "Pt": 0.0,

    # Lanthanides: rough high-spin seeds
    "La": 0.0, "Ce": 1.0, "Pr": 3.0, "Nd": 3.0,
    "Pm": 4.0, "Sm": 5.0, "Eu": 7.0, "Gd": 7.0,
    "Tb": 6.0, "Dy": 5.0, "Ho": 4.0, "Er": 3.0,
    "Tm": 2.0, "Yb": 1.0, "Lu": 0.0,

    # Actinides: rough seeds
    "Ac": 0.0, "Th": 0.0, "Pa": 3.0, "U": 3.0,
    "Np": 4.0, "Pu": 5.0,
}


def parse_poscar_elements_counts(poscar_path: Path):
    lines = poscar_path.read_text(errors="ignore").splitlines()

    if len(lines) < 7:
        raise ValueError("POSCAR too short")

    elements = lines[5].split()
    count_tokens = lines[6].split()

    if all(tok.isdigit() for tok in elements):
        raise ValueError("Old VASP4 POSCAR without element symbols is not supported.")

    counts = [int(x) for x in count_tokens]

    if len(elements) != len(counts):
        raise ValueError(f"Element/count mismatch: {elements} vs {counts}")

    return elements, counts


def parse_outcar_elements_counts(outcar_path: Path):
    lines = outcar_path.read_text(errors="ignore").splitlines()
    elements = []
    counts = None

    for line in lines:
        match = re.search(r"VRHFIN\s*=\s*([A-Z][a-z]?)\s*:", line)
        if match:
            elements.append(match.group(1))
            continue

        if "ions per type" in line and counts is None:
            _, rhs = line.split("=", 1)
            counts = [int(x) for x in rhs.split()]

    if not elements:
        raise ValueError("Could not parse element order from OUTCAR VRHFIN lines.")

    if counts is None:
        raise ValueError("Could not parse 'ions per type' from OUTCAR.")

    if len(elements) != len(counts):
        raise ValueError(f"OUTCAR element/count mismatch: {elements} vs {counts}")

    return elements, counts


def element_count_signature(elements, counts):
    return tuple(zip(elements, counts))


def format_signature(signature):
    return " ".join(f"{element}:{count}" for element, count in signature)


def expand_seed_magmoms(elements, counts, moments, afm=True, unknown_default=0.0):
    values = []

    for element, count in zip(elements, counts):
        base = moments.get(element, unknown_default)

        if abs(base) < 1e-12:
            values.extend([0.0] * count)
            continue

        for i in range(count):
            if afm:
                # Rosen-like alternating pattern: - + - +
                sign = -1.0 if i % 2 == 0 else 1.0
                values.append(sign * base)
            else:
                values.append(base)

    return values


def read_outcar_final_magnetization(outcar_path: Path):
    """
    Parse the final 'magnetization (x)' table from VASP OUTCAR.
    Returns one total magnetic moment per ion.
    """
    lines = outcar_path.read_text(errors="ignore").splitlines()

    block_starts = [
        i for i, line in enumerate(lines)
        if "magnetization (x)" in line
    ]

    if not block_starts:
        raise ValueError("No 'magnetization (x)' block found in OUTCAR.")

    start = block_starts[-1]
    values = []

    for line in lines[start + 1:]:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("-"):
            if values:
                break
            continue

        parts = stripped.split()

        if len(parts) < 5:
            continue

        if not parts[0].isdigit():
            continue

        try:
            values.append(float(parts[-1]))
        except ValueError:
            continue

    if not values:
        raise ValueError("Found magnetization block, but could not parse ion moments.")

    return values


def compress_magmoms(values, decimals=4, zero_tol=5e-5):
    rounded = []

    for v in values:
        if abs(v) < zero_tol:
            v = 0.0
        rounded.append(round(v, decimals))

    parts = []
    current = rounded[0]
    count = 1

    def fmt(x):
        if abs(x) < zero_tol:
            x = 0.0
        return f"{x:.{decimals}f}"

    for v in rounded[1:]:
        if v == current:
            count += 1
        else:
            parts.append(f"{count}*{fmt(current)}")
            current = v
            count = 1

    parts.append(f"{count}*{fmt(current)}")

    return " MAGMOM = " + " ".join(parts)


def remove_existing_magmom(lines):
    output = []
    skipping = False

    for line in lines:
        stripped = line.strip()

        if skipping:
            if stripped.endswith("\\"):
                continue
            skipping = False
            continue

        if stripped.upper().startswith("MAGMOM"):
            if stripped.endswith("\\"):
                skipping = True
            continue

        output.append(line)

    return output


def update_incar(incar_path: Path, magmom_line: str, backup=False):
    lines = incar_path.read_text(errors="ignore").splitlines()

    # Remove any existing MAGMOM line first
    lines = remove_existing_magmom(lines)

    output = []
    inserted = False

    for line in lines:
        output.append(line)

        # Insert MAGMOM immediately after ISPIN
        if line.strip().upper().startswith("ISPIN"):
            output.append(magmom_line)
            inserted = True

    # If no ISPIN line exists, add ISPIN = 2 and then MAGMOM at the end
    if not inserted:
        output.append(" ISPIN = 2")
        output.append(magmom_line)

    if backup:
        shutil.copy2(incar_path, incar_path.with_name(incar_path.name + ".bak"))

    incar_path.write_text("\n".join(output) + "\n")


def parse_overrides(items):
    out = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad override: {item}. Use Element=value, e.g. Cu=1.0")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)

    return out


def seed_mode(args):
    moments = dict(DEFAULT_MOMENTS)
    moments.update(parse_overrides(args.override))

    calc_dirs = []

    for poscar in args.root.rglob("POSCAR"):
        d = poscar.parent

        if args.stage and d.name != args.stage:
            continue

        incar = d / "INCAR"

        if incar.exists():
            calc_dirs.append(d)

    if not calc_dirs:
        print("No folders found with POSCAR and INCAR.")
        return 1

    for d in sorted(calc_dirs):
        poscar = d / "POSCAR"
        incar = d / "INCAR"
        outcar = d / "OUTCAR"

        if outcar.exists():
            print(f"[SKIP] {d}: target OUTCAR exists")
            continue

        try:
            elements, counts = parse_poscar_elements_counts(poscar)
            values = expand_seed_magmoms(
                elements,
                counts,
                moments,
                afm=args.afm,
                unknown_default=args.unknown_default,
            )

            expected = sum(counts)

            if len(values) != expected:
                raise ValueError(f"MAGMOM length {len(values)} != atom count {expected}")

            line = compress_magmoms(values, decimals=args.decimals)

            print(f"[SEED] {d}")
            print(f"       elements = {elements}")
            print(f"       counts   = {counts}")
            print(f"       {line}")

            if args.write:
                update_incar(incar, line, backup=args.backup)
                print("       updated INCAR")
            else:
                print("       dry-run only")

        except Exception as exc:
            print(f"[FAILED] {d}: {exc}")

    return 0


def extract_mode(args):
    mof_dirs = [
        d for d in args.root.iterdir()
        if d.is_dir()
    ]

    if not mof_dirs:
        print("No MOF folders found.")
        return 1

    for mof in sorted(mof_dirs):
        source = mof / args.source_stage
        target = mof / args.target_stage

        source_outcar = source / "OUTCAR"
        source_poscar = source / "POSCAR"
        target_incar = target / "INCAR"
        target_poscar = target / "POSCAR"
        target_outcar = target / "OUTCAR"

        if not source_outcar.exists():
            print(f"[SKIP] {mof.name}: missing {source_outcar}")
            continue

        if target_outcar.exists():
            print(f"[SKIP] {mof.name}: target OUTCAR exists at {target_outcar}")
            continue

        if not source_poscar.exists():
            print(f"[SKIP] {mof.name}: missing {source_poscar}")
            continue

        if not target_incar.exists():
            print(f"[SKIP] {mof.name}: missing {target_incar}")
            continue

        if not target_poscar.exists():
            print(f"[SKIP] {mof.name}: missing {target_poscar}")
            continue

        try:
            source_elements, source_counts = parse_poscar_elements_counts(source_poscar)
            outcar_elements, outcar_counts = parse_outcar_elements_counts(source_outcar)
            target_elements, target_counts = parse_poscar_elements_counts(target_poscar)

            source_signature = element_count_signature(source_elements, source_counts)
            outcar_signature = element_count_signature(outcar_elements, outcar_counts)
            target_signature = element_count_signature(target_elements, target_counts)

            if outcar_signature != source_signature:
                raise ValueError(
                    "source OUTCAR type order does not match source POSCAR: "
                    f"{format_signature(outcar_signature)} != "
                    f"{format_signature(source_signature)}"
                )

            if target_signature != source_signature:
                raise ValueError(
                    "target POSCAR type order does not match source POSCAR: "
                    f"{format_signature(target_signature)} != "
                    f"{format_signature(source_signature)}"
                )

            values = read_outcar_final_magnetization(source_outcar)
            expected = sum(target_counts)

            if len(values) != expected:
                raise ValueError(
                    f"OUTCAR moments {len(values)} != POSCAR atoms {expected}"
                )

            line = compress_magmoms(
                values,
                decimals=args.decimals,
                zero_tol=args.zero_tol,
            )

            print(f"[EXTRACT] {mof.name}")
            print(f"          from: {source_outcar}")
            print(f"          to:   {target_incar}")
            print(f"          verified types: {format_signature(target_signature)}")
            print(f"          {line}")

            if args.write:
                update_incar(target_incar, line, backup=args.backup)
                print("          updated target INCAR")
            else:
                print("          dry-run only")

        except Exception as exc:
            print(f"[FAILED] {mof.name}: {exc}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Manage VASP MAGMOM lines for MOF workflows."
    )

    sub = parser.add_subparsers(dest="mode")

    p_seed = sub.add_parser(
        "seed",
        help="Create initial chemically sensible MAGMOM from POSCAR."
    )
    p_seed.add_argument("--root", required=True, type=Path)
    p_seed.add_argument("--stage", default=None, help="Only folders with this name, e.g. PBED3-PreRelax")
    p_seed.add_argument("--afm", action="store_true", help="Alternate signs for magnetic elements")
    p_seed.add_argument("--unknown-default", type=float, default=0.0)
    p_seed.add_argument("--override", nargs="*", default=[], help="Element overrides, e.g. Cu=1.0 Nd=3.0")
    p_seed.add_argument("--decimals", type=int, default=4)
    p_seed.add_argument("--write", action="store_true")
    p_seed.add_argument("--backup", action="store_true")
    p_seed.set_defaults(func=seed_mode)

    p_ext = sub.add_parser(
        "extract",
        help="Extract final site moments from a source OUTCAR and write MAGMOM to a target INCAR."
    )
    p_ext.add_argument("--root", required=True, type=Path)
    p_ext.add_argument("--source-stage", default="PBED3-Single")
    p_ext.add_argument("--target-stage", default="HSE-single")
    p_ext.add_argument("--decimals", type=int, default=4)
    p_ext.add_argument("--zero-tol", type=float, default=5e-5)
    p_ext.add_argument("--write", action="store_true")
    p_ext.add_argument("--backup", action="store_true")
    p_ext.set_defaults(func=extract_mode)

    args = parser.parse_args()
    if args.mode is None:
        parser.error("the following arguments are required: mode")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
