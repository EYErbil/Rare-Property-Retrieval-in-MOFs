"""
Shared labelled-split loading for figure scripts.

Matches Script 1 (``01_build_inventory_and_master_tables.py``):
  ``{train,val,test}_bandgaps_regression.json``

Each JSON maps MOF id -> HSE06 band gap (eV). Keys use the same
``name`` / ``cif_id`` convention as the PMT embedding NPZ (often with
``_FSR`` suffix).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_FSR_RE = re.compile(r"_FSR$")

SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")
SPLIT_JSON_NAME = "{split}_bandgaps_regression.json"


def strip_id(s: str) -> str:
    s = str(s).strip()
    if s.lower().endswith(".cif"):
        s = s[:-4]
    return s


def id_variants(cid: str) -> List[str]:
    """Return ID variants used to join across files (Script 1 logic)."""
    out: List[str] = []
    s = str(cid).strip()
    if not s:
        return out
    out.append(s)
    bare = s
    if bare.endswith(".cif"):
        bare = bare[:-4]
        out.append(bare)
    if _FSR_RE.search(bare):
        out.append(_FSR_RE.sub("", bare))
    else:
        out.append(bare + "_FSR")
    seen: Set[str] = set()
    dedup: List[str] = []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            dedup.append(v)
    return dedup


def load_split_json(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[str, float] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def load_all_splits(splits_dir: Path,
                    logger: logging.Logger) -> Dict[str, Dict[str, float]]:
    """Load train/val/test JSON splits. Missing files are skipped."""
    out: Dict[str, Dict[str, float]] = {}
    if not splits_dir.exists():
        logger.warning("splits dir not found: %s", splits_dir)
        return out
    for sp in SPLIT_NAMES:
        p = splits_dir / SPLIT_JSON_NAME.format(split=sp)
        if not p.exists():
            logger.warning("split file missing: %s", p)
            continue
        out[sp] = load_split_json(p)
        logger.info("  split %-5s: %d entries  (%s)", sp, len(out[sp]), p.name)
    if not out:
        logger.error("No split JSON files found under %s "
                     "(expected *_bandgaps_regression.json)", splits_dir)
    return out


def labelled_ids_union(splits: Dict[str, Dict[str, float]]) -> Set[str]:
    ids: Set[str] = set()
    for sp in SPLIT_NAMES:
        ids.update(splits.get(sp, {}).keys())
    return ids


def _row_from_split_entry(sid: str,
                          gap: float,
                          split_name: str,
                          hit_threshold_ev: float) -> Dict[str, str]:
    sid = strip_id(sid)
    return {
        "structure_id": sid,
        "true_hse_gap_ev": f"{gap:.6f}",
        "hit_tau_1ev": "1" if gap <= hit_threshold_ev else "0",
        "hse06_dft_done": "1",
        "layer": f"qmof_labelled_{split_name}",
        "split": split_name,
    }


def build_rows_for_splits(
        splits: Dict[str, Dict[str, float]],
        which: Sequence[str] = ("train", "val"),
        hit_threshold_ev: float = 1.0,
) -> List[Dict[str, str]]:
    """Flat list of master-like rows for the requested split names."""
    out: List[Dict[str, str]] = []
    for sp in which:
        for sid, gap in splits.get(sp, {}).items():
            out.append(_row_from_split_entry(sid, gap, sp, hit_threshold_ev))
    return out


def build_rows_by_split(
        splits: Dict[str, Dict[str, float]],
        which: Sequence[str] = ("train", "val"),
        hit_threshold_ev: float = 1.0,
) -> Dict[str, List[Dict[str, str]]]:
    """Per-split dict of master-like rows."""
    out: Dict[str, List[Dict[str, str]]] = {sp: [] for sp in which}
    for sp in which:
        for sid, gap in splits.get(sp, {}).items():
            out[sp].append(_row_from_split_entry(sid, gap, sp, hit_threshold_ev))
    return out


def gaps_from_splits(splits: Dict[str, Dict[str, float]],
                     which: Sequence[str] = SPLIT_NAMES) -> np.ndarray:
    """Concatenate band gaps from the requested splits into one array."""
    vals: List[float] = []
    for sp in which:
        vals.extend(float(v) for v in splits.get(sp, {}).values())
    return np.asarray(vals, dtype=float)


def build_id_index(ids: Sequence[str]) -> Dict[str, int]:
    """Map every variant of every id -> row index (Script 1 logic)."""
    out: Dict[str, int] = {}
    for i, raw in enumerate(ids):
        for v in id_variants(raw):
            out[v] = i
    return out


def lookup_index(cid: str, index: Dict[str, int]) -> Optional[int]:
    for v in id_variants(cid):
        if v in index:
            return index[v]
    return None
