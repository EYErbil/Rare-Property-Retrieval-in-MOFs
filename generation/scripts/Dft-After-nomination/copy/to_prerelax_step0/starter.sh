#!/bin/bash

ROOT="${ROOT:-/path/to/DFT_WORK_ROOT}"
SOURCE_STAGE="${SOURCE_STAGE:-PBED3-Relax}"
TARGET_STAGE="${TARGET_STAGE:-PBED3-PreRelax}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCAR_TEMPLATE="$SCRIPT_DIR/INCAR"

if [ ! -s "$INCAR_TEMPLATE" ]; then
  echo "ERROR: missing INCAR template: $INCAR_TEMPLATE"
  exit 1
fi

copy_required() {
  src="$1"
  dst="$2"

  if [ -s "$src" ]; then
    cp "$src" "$dst"
    echo "  COPIED: $src -> $dst"
  else
    echo "  WARNING missing/empty: $src"
  fi
}

find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name "copy" | sort | while IFS= read -r mofdir; do
  mofname="$(basename "$mofdir")"
  source="$mofdir/$SOURCE_STAGE"
  target="$mofdir/$TARGET_STAGE"

  echo "=============================="
  echo "MOF: $mofname"

  if [ ! -d "$source" ]; then
    echo "  SKIP: no $SOURCE_STAGE folder"
    continue
  fi

  mkdir -p "$target"

  copy_required "$source/POSCAR" "$target/POSCAR"
  copy_required "$source/KPOINTS" "$target/KPOINTS"
  copy_required "$source/POTCAR" "$target/POTCAR"
  copy_required "$source/CHGCAR" "$target/CHGCAR"
  copy_required "$source/WAVECAR" "$target/WAVECAR"

  for shfile in "$source"/*.sh; do
    if [ -f "$shfile" ]; then
      cp "$shfile" "$target/$(basename "$shfile")"
      echo "  COPIED SH: $shfile -> $target/$(basename "$shfile")"
    fi
  done

  if [ -f "$target/INCAR" ]; then
    cp "$target/INCAR" "$target/INCAR.bak_before_prerelax_starter_copy"
  fi

  cp "$INCAR_TEMPLATE" "$target/INCAR"
  echo "  COPIED INCAR TEMPLATE -> $target/INCAR"
done
