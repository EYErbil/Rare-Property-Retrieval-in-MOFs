from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar


def parse_cif_faithfully(cif_path: Path):
    parser = CifParser(
        cif_path,
        occupancy_tolerance=1.0,
        site_tolerance=1e-4,
        frac_tolerance=0,  # do not "fix" fractional coordinates by rounding
    )

    structure = parser.parse_structures(
        primitive=False,  # do not reduce to primitive cell
        check_occu=True,
        on_error="raise",
    )[0]

    return structure


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert all CIF files in a directory to VASP POSCAR files."
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory containing CIF files")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for POSCAR folders")
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write files as output/name.POSCAR instead of output/name/POSCAR",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing POSCAR files",
    )

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    failures = 0

    cif_files = sorted(input_dir.rglob("*.cif"))

    if not cif_files:
        print(f"No CIF files found in {input_dir}")
        return 1

    for cif_path in cif_files:
        name = cif_path.stem

        try:
            structure = parse_cif_faithfully(cif_path)

            if args.flat:
                poscar_path = output_dir / f"{name}.POSCAR"
            else:
                calc_dir = output_dir / name
                calc_dir.mkdir(parents=True, exist_ok=True)
                poscar_path = calc_dir / "POSCAR"

            if poscar_path.exists() and not args.overwrite:
                print(f"[SKIP EXISTS] {poscar_path}")
                report_rows.append({
                    "name": name,
                    "cif": str(cif_path),
                    "poscar": str(poscar_path),
                    "status": "skipped_exists",
                    "formula": str(structure.composition),
                    "num_sites": len(structure),
                    "error": "",
                })
                continue

            Poscar(
                structure,
                sort_structure=False,  # preserve parsed atom order
            ).write_file(
                poscar_path,
                direct=True,  # keep fractional coordinates
            )

            print(f"[OK] {cif_path.name} -> {poscar_path}")

            report_rows.append({
                "name": name,
                "cif": str(cif_path),
                "poscar": str(poscar_path),
                "status": "ok",
                "formula": str(structure.composition),
                "num_sites": len(structure),
                "error": "",
            })

        except Exception as exc:
            failures += 1
            print(f"[FAILED] {cif_path.name}: {exc}")

            report_rows.append({
                "name": name,
                "cif": str(cif_path),
                "poscar": "",
                "status": "failed",
                "formula": "",
                "num_sites": "",
                "error": str(exc),
            })

    report_path = output_dir / "conversion_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "cif",
                "poscar",
                "status",
                "formula",
                "num_sites",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n===== SUMMARY =====")
    print(f"CIF files: {len(cif_files)}")
    print(f"Failures:  {failures}")
    print(f"Report:    {report_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())