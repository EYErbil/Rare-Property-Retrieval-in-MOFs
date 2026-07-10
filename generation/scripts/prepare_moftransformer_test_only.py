#!/usr/bin/env python3
"""
Prepare MOFTransformer data from a CIF folder with TEST-ONLY split.

What this script does:
1) Scans a CIF directory and creates raw_{downstream}.json mapping:
      { "<cif_stem>": <float_value>, ... }
   (example downstream: bandgaps -> raw_bandgaps.json)
2) Calls moftransformer.utils.prepare_data(...) with:
      train_fraction=0.0
      test_fraction=1.0
   so all prepared structures are assigned to test split.

This is intended for generated CIF batches where train/val are not needed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def build_raw_json_from_cifs(
    cif_dir: Path,
    raw_json_path: Path,
    default_value: float,
    random_mode: bool,
    random_min: float,
    random_max: float,
    seed: int,
    overwrite: bool,
) -> dict[str, float]:
    if raw_json_path.exists() and not overwrite:
        with raw_json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        print(f"Using existing JSON: {raw_json_path} ({len(data)} entries)")
        return {str(k): float(v) for k, v in data.items()}

    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        raise ValueError(f"No .cif files found in {cif_dir}")

    rng = random.Random(seed)
    data: dict[str, float] = {}
    for p in cif_files:
        if random_mode:
            value = rng.uniform(random_min, random_max)
        else:
            value = default_value
        data[p.stem] = float(value)

    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_json_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    mode_msg = (
        f"random uniform in [{random_min}, {random_max}]"
        if random_mode
        else f"constant {default_value}"
    )
    print(
        f"Created {raw_json_path} with {len(data)} entries "
        f"using {mode_msg}."
    )
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate raw_{downstream}.json from CIF names and run "
            "moftransformer prepare_data with test_fraction=1.0"
        )
    )
    parser.add_argument(
        "--cif-dir",
        required=True,
        type=Path,
        help="Directory containing generated .cif files.",
    )
    parser.add_argument(
        "--output-dataset-dir",
        required=True,
        type=Path,
        help="Empty or existing destination directory for prepared outputs.",
    )
    parser.add_argument(
        "--downstream",
        default="bandgaps",
        help="Downstream task name. Default: bandgaps",
    )
    parser.add_argument(
        "--raw-json-path",
        default=None,
        type=Path,
        help=(
            "Optional explicit path for raw_{downstream}.json. "
            "Default: <cif-dir>/raw_<downstream>.json"
        ),
    )
    parser.add_argument(
        "--overwrite-raw-json",
        action="store_true",
        help="Overwrite existing raw_{downstream}.json if present.",
    )
    parser.add_argument(
        "--default-target-value",
        type=float,
        default=0.0,
        help=(
            "Value written for each CIF in raw JSON when not using random mode. "
            "Default: 0.0"
        ),
    )
    parser.add_argument(
        "--random-target-values",
        action="store_true",
        help=(
            "If set, assign random float values per CIF instead of a constant. "
            "Useful when a non-constant placeholder JSON is needed."
        ),
    )
    parser.add_argument(
        "--random-min",
        type=float,
        default=0.0,
        help="Minimum random target value (used only with --random-target-values).",
    )
    parser.add_argument(
        "--random-max",
        type=float,
        default=8.0,
        help="Maximum random target value (used only with --random-target-values).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random target values and split internals.",
    )
    parser.add_argument(
        "--max-num-unique-atoms",
        type=int,
        default=300,
        help="Forwarded to prepare_data().",
    )
    parser.add_argument(
        "--max-num-supercell-atoms",
        type=int,
        default=None,
        help="Forwarded to prepare_data() as max_num_atoms.",
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=60.0,
        help="Forwarded to prepare_data().",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=30.0,
        help="Forwarded to prepare_data().",
    )
    parser.add_argument(
        "--max-num-nbr",
        type=int,
        default=12,
        help="Forwarded to prepare_data().",
    )
    args = parser.parse_args()

    cif_dir = args.cif_dir.resolve()
    out_dir = args.output_dataset_dir.resolve()
    if not cif_dir.is_dir():
        raise ValueError(f"--cif-dir is not a directory: {cif_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.random_target_values and args.random_min > args.random_max:
        raise ValueError("--random-min cannot be greater than --random-max")

    raw_json_path = (
        args.raw_json_path.resolve()
        if args.raw_json_path is not None
        else (cif_dir / f"raw_{args.downstream}.json")
    )

    _ = build_raw_json_from_cifs(
        cif_dir=cif_dir,
        raw_json_path=raw_json_path,
        default_value=args.default_target_value,
        random_mode=args.random_target_values,
        random_min=args.random_min,
        random_max=args.random_max,
        seed=args.seed,
        overwrite=args.overwrite_raw_json,
    )

    try:
        from moftransformer.utils import prepare_data
    except ImportError as e:
        print(
            "Failed to import moftransformer.utils.prepare_data.\n"
            "Activate the correct environment and ensure moftransformer is installed.",
            file=sys.stderr,
        )
        raise e

    print("Running prepare_data(...) with TEST-ONLY split:")
    print(f"  root_cifs      = {cif_dir}")
    print(f"  root_dataset   = {out_dir}")
    print(f"  downstream     = {args.downstream}")
    print("  train_fraction = 0.0")
    print("  test_fraction  = 1.0")

    prepare_data(
        str(cif_dir),
        str(out_dir),
        downstream=args.downstream,
        seed=args.seed,
        train_fraction=0.0,
        test_fraction=1.0,
        max_num_unique_atoms=args.max_num_unique_atoms,
        max_num_atoms=args.max_num_supercell_atoms,
        max_length=args.max_length,
        min_length=args.min_length,
        max_num_nbr=args.max_num_nbr,
    )

    print("Done.")
    print(f"Raw JSON: {raw_json_path}")
    print(f"Prepared dataset root: {out_dir}")
    print("Expected populated split directory: test/")
    print("Expected near-empty split directories: train/ and val/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

