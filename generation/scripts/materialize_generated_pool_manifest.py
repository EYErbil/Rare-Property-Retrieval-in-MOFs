#!/usr/bin/env python3
'''Materialize generated-pool IDs from a descriptor archive.

The released generated SOAP archive stores one cif_ids value per descriptor
row. This utility writes that authoritative array as a newline-delimited
manifest, preserving descriptor-row order and requiring no separate CSV.
'''

import argparse
import hashlib
from pathlib import Path

import numpy as np


PAPER_POOL_SIZE = 13_802


def _as_identifier(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        identifier = bytes(value).decode('utf-8')
    elif isinstance(value, (str, np.str_)):
        identifier = str(value)
    else:
        raise ValueError(
            'cif_ids must contain strings, not '
            f'{type(value).__name__} values'
        )
    if not identifier:
        raise ValueError('cif_ids contains an empty identifier')
    if identifier != identifier.strip():
        raise ValueError(f'identifier has surrounding whitespace: {identifier!r}')
    if '\n' in identifier or '\r' in identifier:
        raise ValueError(f'identifier contains a newline: {identifier!r}')
    return identifier


def load_identifiers(
    descriptor_npz: Path,
    *,
    id_key: str = 'cif_ids',
    expected_count: int = PAPER_POOL_SIZE,
) -> list[str]:
    '''Load unique IDs while preserving their descriptor-row order.'''
    with np.load(descriptor_npz, allow_pickle=False) as archive:
        if id_key not in archive.files:
            available = ', '.join(archive.files)
            raise ValueError(
                f'{descriptor_npz} has no {id_key!r} array; keys: {available}'
            )
        raw_ids = np.asarray(archive[id_key])

    if raw_ids.ndim != 1:
        raise ValueError(f'{id_key} must be 1D, got shape {raw_ids.shape}')
    identifiers = [_as_identifier(value) for value in raw_ids]
    if expected_count > 0 and len(identifiers) != expected_count:
        raise ValueError(
            f'expected {expected_count:,} IDs, found {len(identifiers):,} '
            f'in {descriptor_npz}'
        )
    unique_count = len(set(identifiers))
    if unique_count != len(identifiers):
        raise ValueError(
            f'{id_key} contains {len(identifiers) - unique_count:,} duplicate row(s)'
        )
    return identifiers


def materialize_manifest(
    descriptor_npz: Path,
    output: Path,
    *,
    id_key: str = 'cif_ids',
    expected_count: int = PAPER_POOL_SIZE,
) -> tuple[int, str]:
    '''Write an ordered text manifest and return its row count and SHA-256.'''
    identifiers = load_identifiers(
        descriptor_npz, id_key=id_key, expected_count=expected_count
    )
    payload = ''.join(f'{identifier}\n' for identifier in identifiers).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return len(identifiers), hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Materialize the exact generated-pool ID manifest from an NPZ '
            'archive while preserving descriptor-row order.'
        )
    )
    parser.add_argument(
        '--descriptor-npz',
        type=Path,
        required=True,
        help='NPZ containing the generated-pool ID array.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Destination newline-delimited manifest.',
    )
    parser.add_argument(
        '--id-key',
        default='cif_ids',
        help='Identifier-array key (default: cif_ids).',
    )
    parser.add_argument(
        '--expected-count',
        type=int,
        default=PAPER_POOL_SIZE,
        help=(
            f'Required row count (default: {PAPER_POOL_SIZE}); use 0 to '
            'disable the paper-size check.'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count, digest = materialize_manifest(
        args.descriptor_npz,
        args.output,
        id_key=args.id_key,
        expected_count=args.expected_count,
    )
    print(f'Wrote {count:,} unique IDs to {args.output}')
    print(f'Manifest SHA-256: {digest}')


if __name__ == '__main__':
    main()
