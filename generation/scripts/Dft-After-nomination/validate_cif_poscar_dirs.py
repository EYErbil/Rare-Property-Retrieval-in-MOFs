from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar


def parse_cif_faithfully(cif_path: Path):
    parser = CifParser(
        cif_path,
        occupancy_tolerance=1.0,
        site_tolerance=1e-4,
        frac_tolerance=0,
    )

    structure = parser.parse_structures(
        primitive=False,
        check_occu=True,
        on_error="raise",
    )[0]

    return structure


def find_poscar(poscar_root: Path, name: str, flat: bool) -> Path | None:
    if flat:
        candidates = [
            poscar_root / f"{name}.POSCAR",
            poscar_root / f"{name}.vasp",
            poscar_root / name,
        ]
    else:
        candidates = [
            poscar_root / name / "POSCAR",
            poscar_root / name / f"{name}.POSCAR",
            poscar_root / f"{name}.POSCAR",
        ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    return None


def max_frac_coord_diff(a, b) -> float:
    """
    Compare fractional coordinates with periodic wrapping.
    Assumes atom order is preserved.
    """
    fa = np.array(a.frac_coords)
    fb = np.array(b.frac_coords)

    diff = fa - fb
    diff -= np.round(diff)  # minimum-image fractional difference

    return float(np.max(np.abs(diff)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate CIF files against generated POSCAR files name-by-name."
    )
    parser.add_argument("--cifs", required=True, type=Path, help="Directory containing CIF files")
    parser.add_argument("--poscars", required=True, type=Path, help="Directory containing POSCAR outputs")
    parser.add_argument("--report", required=True, type=Path, help="CSV report output path")
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Use if POSCARs are stored as poscars/name.POSCAR instead of poscars/name/POSCAR",
    )
    parser.add_argument(
        "--lattice-tol",
        type=float,
        default=1e-5,
        help="Tolerance for lattice length/angle differences",
    )
    parser.add_argument(
        "--coord-tol",
        type=float,
        default=1e-5,
        help="Tolerance for fractional coordinate differences",
    )

    args = parser.parse_args()

    cif_files = sorted(args.cifs.rglob("*.cif"))

    if not cif_files:
        print(f"No CIF files found in {args.cifs}")
        return 1

    rows = []
    failures = 0

    for cif_path in cif_files:
        name = cif_path.stem
        poscar_path = find_poscar(args.poscars, name, args.flat)

        row = {
            "name": name,
            "cif": str(cif_path),
            "poscar": str(poscar_path) if poscar_path else "",
            "status": "",
            "same_formula": "",
            "same_num_sites": "",
            "max_lattice_abc_diff": "",
            "max_lattice_angle_diff": "",
            "max_frac_coord_diff": "",
            "cif_formula": "",
            "poscar_formula": "",
            "cif_num_sites": "",
            "poscar_num_sites": "",
            "error": "",
        }

        try:
            if poscar_path is None:
                row["status"] = "missing_poscar"
                row["error"] = "No matching POSCAR found"
                failures += 1
                rows.append(row)
                print(f"[MISSING POSCAR] {name}")
                continue

            cif_structure = parse_cif_faithfully(cif_path)
            poscar_structure = Poscar.from_file(poscar_path).structure

            row["cif_formula"] = str(cif_structure.composition)
            row["poscar_formula"] = str(poscar_structure.composition)
            row["cif_num_sites"] = len(cif_structure)
            row["poscar_num_sites"] = len(poscar_structure)

            same_formula = cif_structure.composition == poscar_structure.composition
            same_num_sites = len(cif_structure) == len(poscar_structure)

            abc_diff = np.max(
                np.abs(
                    np.array(cif_structure.lattice.abc)
                    - np.array(poscar_structure.lattice.abc)
                )
            )

            angle_diff = np.max(
                np.abs(
                    np.array(cif_structure.lattice.angles)
                    - np.array(poscar_structure.lattice.angles)
                )
            )

            row["same_formula"] = same_formula
            row["same_num_sites"] = same_num_sites
            row["max_lattice_abc_diff"] = float(abc_diff)
            row["max_lattice_angle_diff"] = float(angle_diff)

            if same_num_sites:
                coord_diff = max_frac_coord_diff(cif_structure, poscar_structure)
                row["max_frac_coord_diff"] = coord_diff
            else:
                coord_diff = float("inf")
                row["max_frac_coord_diff"] = "not_checked_num_sites_differ"

            passed = (
                same_formula
                and same_num_sites
                and abc_diff <= args.lattice_tol
                and angle_diff <= args.lattice_tol
                and coord_diff <= args.coord_tol
            )

            if passed:
                row["status"] = "pass"
                print(f"[PASS] {name}")
            else:
                row["status"] = "fail"
                failures += 1
                print(f"[FAIL] {name}")

        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            failures += 1
            print(f"[ERROR] {name}: {exc}")

        rows.append(row)

    args.report.parent.mkdir(parents=True, exist_ok=True)

    with args.report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "cif",
                "poscar",
                "status",
                "same_formula",
                "same_num_sites",
                "max_lattice_abc_diff",
                "max_lattice_angle_diff",
                "max_frac_coord_diff",
                "cif_formula",
                "poscar_formula",
                "cif_num_sites",
                "poscar_num_sites",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== SUMMARY =====")
    print(f"Checked:  {len(cif_files)}")
    print(f"Failures: {failures}")
    print(f"Report:   {args.report}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())