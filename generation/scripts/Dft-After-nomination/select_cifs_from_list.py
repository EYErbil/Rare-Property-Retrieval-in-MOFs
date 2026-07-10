from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def read_names(txt_file: Path) -> list[str]:
    names = []
    for line in txt_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().endswith(".cif"):
            line = Path(line).stem
        names.append(line)
    return names


def find_matching_cif(source_dir: Path, target_name: str) -> list[Path]:
    all_cifs = list(source_dir.rglob("*.cif"))

    # First: exact stem match
    exact = [p for p in all_cifs if p.stem == target_name]
    if exact:
        return exact

    # Second: filename contains target name
    partial = [p for p in all_cifs if target_name in p.stem]
    return partial


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy selected CIF files from a source directory using names from a .txt file."
    )
    parser.add_argument("--source", required=True, type=Path, help="Directory containing CIF files")
    parser.add_argument("--list", required=True, type=Path, help="Text file with CIF names/stems")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for copied CIFs")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CIFs in output directory",
    )

    args = parser.parse_args()

    source_dir = args.source
    list_file = args.list
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    names = read_names(list_file)

    copied = 0
    missing = []
    ambiguous = []

    for name in names:
        matches = find_matching_cif(source_dir, name)

        if len(matches) == 0:
            missing.append(name)
            print(f"[MISSING] {name}")
            continue

        if len(matches) > 1:
            ambiguous.append((name, matches))
            print(f"[AMBIGUOUS] {name}")
            for m in matches:
                print(f"    {m}")
            continue

        src = matches[0]
        dst = output_dir / src.name

        if dst.exists() and not args.overwrite:
            print(f"[SKIP EXISTS] {dst}")
            continue

        shutil.copy2(src, dst)
        copied += 1
        print(f"[COPIED] {src.name} -> {dst}")

    print("\n===== SUMMARY =====")
    print(f"Requested: {len(names)}")
    print(f"Copied:    {copied}")
    print(f"Missing:   {len(missing)}")
    print(f"Ambiguous: {len(ambiguous)}")

    if missing:
        print("\nMissing names:")
        for m in missing:
            print(f"  {m}")

    if ambiguous:
        print("\nAmbiguous names:")
        for name, matches in ambiguous:
            print(f"  {name}")
            for m in matches:
                print(f"    {m}")

    return 1 if missing or ambiguous else 0


if __name__ == "__main__":
    raise SystemExit(main())