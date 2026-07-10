import argparse
import csv
import math
from pathlib import Path

from pymatgen.io.vasp import Poscar


def choose_kmesh(structure, kppa=500, max_grid=30):
    """
    Rosen / QMOF / Materials-Project-MOF-like KPOINTS rule.

    Target:
        k1 * k2 * k3 >= ceil(kppa / natoms)

    Shape:
        k_i follows reciprocal lattice vector lengths.

    This matches the logic:
        larger real-space cell direction -> smaller k_i
        shorter real-space cell direction -> larger k_i

    Examples:
        40 atoms, compact anisotropic cell -> often 4 2 2
        58 atoms, long c direction         -> often 3 3 1
    """

    natoms = len(structure)
    target = max(1, int(math.ceil(float(kppa) / float(natoms))))

    recip = structure.lattice.reciprocal_lattice.abc
    b1, b2, b3 = recip

    geom_mean = (b1 * b2 * b3) ** (1.0 / 3.0)
    scale = target ** (1.0 / 3.0)

    ideal = [
        max(1.0, scale * b1 / geom_mean),
        max(1.0, scale * b2 / geom_mean),
        max(1.0, scale * b3 / geom_mean),
    ]

    best_mesh = None
    best_score = None

    for k1 in range(1, max_grid + 1):
        for k2 in range(1, max_grid + 1):
            for k3 in range(1, max_grid + 1):
                product = k1 * k2 * k3

                if product < target:
                    continue

                mesh = [k1, k2, k3]

                product_penalty = float(product - target) / float(target)

                shape_penalty = 0.0
                for k, x in zip(mesh, ideal):
                    shape_penalty += math.log(float(k) / float(x)) ** 2

                # Product closeness matters most.
                # Shape closeness follows reciprocal lattice proportions.
                score = product_penalty + 0.35 * shape_penalty

                if best_score is None or score < best_score:
                    best_score = score
                    best_mesh = mesh

    return best_mesh, target, ideal, recip


def choose_style(mesh):
    """
    Style heuristic matching your Rosen examples:

        4 2 2 -> Monkhorst-Pack
        3 3 1 -> Gamma

    Rule:
        if any direction is 1, use Gamma
        else if all directions are even, use Monkhorst-Pack
        else use Gamma

    This keeps Gamma included for odd or low-dimensional meshes.
    """

    if 1 in mesh:
        return "Gamma"

    if all(k % 2 == 0 for k in mesh):
        return "Monkhorst-Pack"

    return "Gamma"


def write_kpoints(path, mesh, style):
    text = (
        "KPOINTS created by Rosen-like KPPRA script\n"
        "0\n"
        f"{style}\n"
        f"{mesh[0]} {mesh[1]} {mesh[2]}\n"
        "0 0 0\n"
    )

    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(
        description="Create Rosen/QMOF-like KPOINTS files from POSCAR folders."
    )

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory containing VASP folders with POSCAR files."
    )

    parser.add_argument(
        "--kppa",
        type=int,
        default=500,
        help="KPPRA-like value. Use 500 for Rosen/QMOF-like MOF HSE screening."
    )

    parser.add_argument(
        "--style",
        choices=["auto", "Gamma", "Monkhorst-Pack"],
        default="auto",
        help="KPOINTS style. Default auto matches the Rosen-like examples."
    )

    parser.add_argument(
        "--max-grid",
        type=int,
        default=30,
        help="Maximum allowed k-grid value in each direction."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing KPOINTS files."
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional CSV report path."
    )

    args = parser.parse_args()

    poscars = sorted(args.root.rglob("POSCAR"))

    if not poscars:
        print(f"No POSCAR files found under {args.root}")
        return 1

    rows = []
    written = 0
    skipped = 0
    failed = 0

    for poscar in poscars:
        folder = poscar.parent
        kpoints_path = folder / "KPOINTS"

        if kpoints_path.exists() and not args.overwrite:
            print(f"[SKIP EXISTS] {kpoints_path}")
            skipped += 1
            continue

        try:
            structure = Poscar.from_file(poscar).structure

            mesh, target, ideal, recip = choose_kmesh(
                structure,
                kppa=args.kppa,
                max_grid=args.max_grid,
            )

            if args.style == "auto":
                style = choose_style(mesh)
            else:
                style = args.style

            write_kpoints(kpoints_path, mesh, style)

            product = mesh[0] * mesh[1] * mesh[2]

            print(
                f"[OK] {folder.name}: {style} "
                f"{mesh[0]} {mesh[1]} {mesh[2]} "
                f"(atoms={len(structure)}, target={target}, product={product})"
            )

            rows.append({
                "folder": str(folder),
                "formula": str(structure.composition),
                "natoms": len(structure),
                "kppa": args.kppa,
                "target_product": target,
                "k1": mesh[0],
                "k2": mesh[1],
                "k3": mesh[2],
                "product": product,
                "style": style,
                "a": structure.lattice.a,
                "b": structure.lattice.b,
                "c": structure.lattice.c,
                "alpha": structure.lattice.alpha,
                "beta": structure.lattice.beta,
                "gamma": structure.lattice.gamma,
                "reciprocal_b1": recip[0],
                "reciprocal_b2": recip[1],
                "reciprocal_b3": recip[2],
                "ideal_k1": ideal[0],
                "ideal_k2": ideal[1],
                "ideal_k3": ideal[2],
                "status": "ok",
                "error": "",
            })

            written += 1

        except Exception as exc:
            print(f"[FAILED] {poscar}: {exc}")
            failed += 1

            rows.append({
                "folder": str(folder),
                "formula": "",
                "natoms": "",
                "kppa": args.kppa,
                "target_product": "",
                "k1": "",
                "k2": "",
                "k3": "",
                "product": "",
                "style": "",
                "a": "",
                "b": "",
                "c": "",
                "alpha": "",
                "beta": "",
                "gamma": "",
                "reciprocal_b1": "",
                "reciprocal_b2": "",
                "reciprocal_b3": "",
                "ideal_k1": "",
                "ideal_k2": "",
                "ideal_k3": "",
                "status": "failed",
                "error": str(exc),
            })

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)

        with args.report.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "folder",
                "formula",
                "natoms",
                "kppa",
                "target_product",
                "k1",
                "k2",
                "k3",
                "product",
                "style",
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
                "reciprocal_b1",
                "reciprocal_b2",
                "reciprocal_b3",
                "ideal_k1",
                "ideal_k2",
                "ideal_k3",
                "status",
                "error",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\n===== SUMMARY =====")
    print(f"POSCARs found: {len(poscars)}")
    print(f"KPOINTS written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

    if args.report is not None:
        print(f"Report: {args.report}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())