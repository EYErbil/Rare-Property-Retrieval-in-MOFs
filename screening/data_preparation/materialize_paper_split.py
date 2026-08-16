#!/usr/bin/env python3
"""Restore and verify the exact QMOF partition used in the paper.

The labeled pretrained-PMTransformer archive is the authoritative record of
partition membership. This script materializes only the three regression JSON
files required downstream and deliberately refuses to overwrite any existing
file. Existing files are instead checked against the archive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


SPLIT_NAMES = ("train", "val", "test")
PAPER_TOTALS = {"train": 1136, "val": 524, "test": 9150}
PAPER_POSITIVES = {"train": 60, "val": 5, "test": 9}
PAPER_THRESHOLD_EV = 1.0
REQUIRED_ARCHIVE_KEYS = frozenset({"cif_ids", "bandgaps", "splits"})


class SplitValidationError(ValueError):
    """Raised when an archive or existing split is not the paper partition."""


def _text(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def load_and_validate_archive(
    archive_path: Path,
    *,
    expected_totals: Mapping[str, int] = PAPER_TOTALS,
    expected_positives: Mapping[str, int] = PAPER_POSITIVES,
    threshold_ev: float = PAPER_THRESHOLD_EV,
) -> dict[str, dict[str, float]]:
    """Load partition assignments and enforce all paper invariants."""

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise SplitValidationError(f"archive does not exist: {archive_path}")

    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            missing = REQUIRED_ARCHIVE_KEYS.difference(archive.files)
            if missing:
                raise SplitValidationError(
                    "archive is missing required keys: " + ", ".join(sorted(missing))
                )

            cif_ids_raw = np.asarray(archive["cif_ids"])
            bandgaps_raw = np.asarray(archive["bandgaps"])
            splits_raw = np.asarray(archive["splits"])

            if "embeddings" in archive.files:
                embeddings = archive["embeddings"]
                if embeddings.ndim != 2 or embeddings.shape[0] != len(cif_ids_raw):
                    raise SplitValidationError(
                        "embeddings must be a 2-D array with one row per cif_id"
                    )
    except SplitValidationError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise SplitValidationError(f"cannot read archive {archive_path}: {exc}") from exc

    arrays = {
        "cif_ids": cif_ids_raw,
        "bandgaps": bandgaps_raw,
        "splits": splits_raw,
    }
    for name, array in arrays.items():
        if array.ndim != 1:
            raise SplitValidationError(f"{name} must be one-dimensional, got {array.shape}")

    n_rows = len(cif_ids_raw)
    if len(bandgaps_raw) != n_rows or len(splits_raw) != n_rows:
        raise SplitValidationError(
            "cif_ids, bandgaps, and splits must have the same number of rows"
        )

    cif_ids = [_text(value) for value in cif_ids_raw]
    split_labels = [_text(value).strip().lower() for value in splits_raw]
    try:
        bandgaps = np.asarray(bandgaps_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SplitValidationError(f"bandgaps are not numeric: {exc}") from exc

    if any(not cif_id for cif_id in cif_ids):
        raise SplitValidationError("cif_ids contains an empty identifier")
    if len(set(cif_ids)) != n_rows:
        raise SplitValidationError("cif_ids must be globally unique")
    if not np.isfinite(bandgaps).all():
        raise SplitValidationError("bandgaps contains a non-finite value")

    observed_labels = set(split_labels)
    expected_labels = set(SPLIT_NAMES)
    if observed_labels != expected_labels:
        raise SplitValidationError(
            f"split labels must be {sorted(expected_labels)}, got {sorted(observed_labels)}"
        )

    boundary_count = int(np.count_nonzero(bandgaps == threshold_ev))
    if boundary_count:
        raise SplitValidationError(
            f"archive contains {boundary_count} bandgap value(s) exactly at "
            f"the {threshold_ev:g} eV decision boundary"
        )

    materialized: dict[str, dict[str, float]] = {
        split_name: {} for split_name in SPLIT_NAMES
    }
    for cif_id, bandgap, split_name in zip(cif_ids, bandgaps, split_labels):
        materialized[split_name][cif_id] = float(bandgap)

    for split_name in SPLIT_NAMES:
        expected_total = expected_totals[split_name]
        observed_total = len(materialized[split_name])
        if observed_total != expected_total:
            raise SplitValidationError(
                f"{split_name} has {observed_total} rows; expected {expected_total}"
            )

        # The paper definition is inclusive. The archive has no values exactly
        # at the boundary, which is checked above.
        observed_positive = sum(
            bandgap <= threshold_ev for bandgap in materialized[split_name].values()
        )
        expected_positive = expected_positives[split_name]
        if observed_positive != expected_positive:
            raise SplitValidationError(
                f"{split_name} has {observed_positive} positives at <= "
                f"{threshold_ev:g} eV; expected {expected_positive}"
            )

    if sum(map(len, materialized.values())) != n_rows:
        raise SplitValidationError("not every archive row was assigned exactly once")

    return materialized


def _load_existing(path: Path) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SplitValidationError(f"cannot read existing split {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SplitValidationError(f"existing split is not a JSON object: {path}")

    normalized: dict[str, float] = {}
    for cif_id, bandgap in payload.items():
        try:
            value = float(bandgap)
        except (TypeError, ValueError) as exc:
            raise SplitValidationError(
                f"existing split has a non-numeric bandgap for {cif_id!r}: {path}"
            ) from exc
        if not math.isfinite(value):
            raise SplitValidationError(
                f"existing split has a non-finite bandgap for {cif_id!r}: {path}"
            )
        normalized[str(cif_id)] = value
    return normalized


def _describe_mismatch(
    expected: Mapping[str, float], observed: Mapping[str, float]
) -> str:
    missing = sorted(set(expected).difference(observed))
    extra = sorted(set(observed).difference(expected))
    changed = sorted(
        cif_id
        for cif_id in set(expected).intersection(observed)
        if expected[cif_id] != observed[cif_id]
    )
    parts = []
    if missing:
        parts.append(f"missing {len(missing)} ID(s), first={missing[0]!r}")
    if extra:
        parts.append(f"extra {len(extra)} ID(s), first={extra[0]!r}")
    if changed:
        parts.append(f"changed {len(changed)} value(s), first={changed[0]!r}")
    return "; ".join(parts) or "content differs"


def materialize_or_verify(
    split_data: Mapping[str, Mapping[str, float]],
    output_dir: Path,
    *,
    verify_only: bool = False,
) -> dict[str, str]:
    """Create missing files and verify existing files without overwriting."""

    output_dir = Path(output_dir)
    if verify_only:
        if not output_dir.is_dir():
            raise SplitValidationError(f"split directory does not exist: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        split_name: output_dir / f"{split_name}_bandgaps_regression.json"
        for split_name in SPLIT_NAMES
    }

    # Preflight every existing file before creating anything, so a stale file
    # cannot leave behind a partially materialized canonical partition.
    states: dict[str, str] = {}
    for split_name, path in targets.items():
        if not path.exists():
            if verify_only:
                raise SplitValidationError(f"required split file is missing: {path}")
            states[split_name] = "missing"
            continue
        if not path.is_file():
            raise SplitValidationError(f"split target is not a regular file: {path}")
        observed = _load_existing(path)
        expected = dict(split_data[split_name])
        if observed != expected:
            raise SplitValidationError(
                f"refusing to overwrite noncanonical split {path}: "
                + _describe_mismatch(expected, observed)
            )
        states[split_name] = "verified"

    for split_name, path in targets.items():
        if states[split_name] != "missing":
            continue
        serialized = json.dumps(
            dict(split_data[split_name]), indent=2, ensure_ascii=False
        ) + "\n"
        try:
            # Exclusive creation is intentional: even a concurrent invocation
            # is not allowed to replace the canonical file.
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
        except FileExistsError:
            observed = _load_existing(path)
            expected = dict(split_data[split_name])
            if observed != expected:
                raise SplitValidationError(
                    f"split appeared concurrently with different content: {path}"
                )
            states[split_name] = "verified"
        else:
            states[split_name] = "created"

    return states


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize or verify the exact paper split from its labeled "
        "pretrained-PMTransformer archive."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="require all canonical JSONs to exist; do not create missing files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        split_data = load_and_validate_archive(args.archive)
        states = materialize_or_verify(
            split_data, args.output_dir, verify_only=args.verify_only
        )
    except (OSError, SplitValidationError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    print(f"Verified paper threshold: bandgap <= {PAPER_THRESHOLD_EV:g} eV")
    for split_name in SPLIT_NAMES:
        print(
            f"  {split_name}: {PAPER_TOTALS[split_name]} total, "
            f"{PAPER_POSITIVES[split_name]} positive, {states[split_name]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
