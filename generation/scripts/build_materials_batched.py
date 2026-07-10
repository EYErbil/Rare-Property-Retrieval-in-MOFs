"""
Run ``bulk_pormake_generation/build_materials.py`` in **chunks** of
candidates, each in a **new Python process**.

Why: one process that loops thousands of ``build_by_type`` calls can grow RAM
(fragmentation, caches, large intermediates) until the job hits OOM even on
high-memory nodes. Fresh processes reset the heap between chunks.

Usage (cluster, same args as ``build_materials.py``)::

    python scripts/build_materials_batched.py \\
        --candidates data/candidates_qmof_13k_200atom.txt \\
        --bb-dir qmof_bb_dir \\
        --topo-dir qmof_topo_dir \\
        --save-dir generated_cifs/small_30A_200atom \\
        --large-dir generated_cifs/large_30A_200atom \\
        --cutoff 30.0 \\
        --chunk-size 400

Resume: by default, names that already have a ``.cif`` in either output
directory are skipped (use ``--no-skip-existing`` to force rebuild).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BUILD_SCRIPT = PROJECT_ROOT / "bulk_pormake_generation" / "build_materials.py"


def _existing_cif_names(save_dir: Path, large_dir: Path) -> set[str]:
    names: set[str] = set()
    for d in (save_dir, large_dir):
        if not d.is_dir():
            continue
        for p in d.glob("*.cif"):
            names.add(p.stem)
    return names


def _chunked(names: list[str], size: int) -> list[list[str]]:
    return [names[i : i + size] for i in range(0, len(names), size)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch wrapper: subprocess per chunk for stable memory use."
    )
    parser.add_argument(
        "-c",
        "--candidates",
        "--candidate-file",
        required=True,
        type=Path,
        help="Whitespace-separated MOF name list (same as build_materials.py).",
    )
    parser.add_argument("-b", "--bb-dir", required=True, type=Path)
    parser.add_argument("-t", "--topo-dir", required=True, type=Path)
    parser.add_argument("-s", "--save-dir", type=Path, default=Path("small/"))
    parser.add_argument("-l", "--large-dir", type=Path, default=Path("large/"))
    parser.add_argument("-co", "--cutoff", type=float, default=60.0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Candidates per subprocess (default 400). Lower if a single chunk still OOMs.",
    )
    parser.add_argument(
        "--build-script",
        type=Path,
        default=DEFAULT_BUILD_SCRIPT,
        help="Path to build_materials.py",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip names that already have .cif in save-dir or large-dir (default: skip).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining chunks after a non-zero exit (default: stop on first failure).",
    )
    args = parser.parse_args()

    build_script = args.build_script.resolve()
    if not build_script.is_file():
        print(f"build script not found: {build_script}", file=sys.stderr)
        return 2

    text = args.candidates.read_text(encoding="utf-8", errors="replace")
    names = [w for w in text.split() if w]
    if not names:
        print("No candidate names in file.", file=sys.stderr)
        return 1

    save_dir = args.save_dir.resolve()
    large_dir = args.large_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    large_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        done = _existing_cif_names(save_dir, large_dir)
        before = len(names)
        names = [n for n in names if n not in done]
        skipped = before - len(names)
        if skipped:
            print(f"Skipping {skipped} names with existing .cif under outputs.")
    if not names:
        print("Nothing left to build.")
        return 0

    chunks = _chunked(names, max(1, args.chunk_size))
    print(f"Building {len(names)} candidates in {len(chunks)} chunk(s), chunk_size={args.chunk_size}.")

    with tempfile.TemporaryDirectory(prefix="pormake_chunks_") as tmp:
        tmp_path = Path(tmp)
        for i, chunk in enumerate(chunks):
            chunk_file = tmp_path / f"chunk_{i:04d}.txt"
            chunk_file.write_text(" ".join(chunk) + "\n", encoding="utf-8")
            cmd = [
                sys.executable,
                str(build_script),
                "-c",
                str(chunk_file),
                "-b",
                str(args.bb_dir.resolve()),
                "-t",
                str(args.topo_dir.resolve()),
                "-s",
                str(save_dir),
                "-l",
                str(large_dir),
                "-co",
                str(args.cutoff),
            ]
            print(f"\n--- chunk {i + 1}/{len(chunks)} ({len(chunk)} names) ---")
            print(" ", " ".join(cmd))
            r = subprocess.run(cmd, check=False)
            if r.returncode != 0:
                print(
                    f"Chunk {i + 1} exited with code {r.returncode}.",
                    file=sys.stderr,
                )
                if not args.continue_on_error:
                    return r.returncode

    print("\nAll chunks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
