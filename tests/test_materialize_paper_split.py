import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "screening"
    / "data_preparation"
    / "materialize_paper_split.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_paper_split", SCRIPT_PATH)
PAPER_SPLIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAPER_SPLIT)


SMALL_TOTALS = {"train": 3, "val": 2, "test": 2}
SMALL_POSITIVES = {"train": 1, "val": 1, "test": 1}
SMALL_ROWS = [
    ("train-low", 0.50, "train"),
    ("train-high-a", 1.10, "train"),
    ("train-high-b", 3.20, "train"),
    ("val-low", 0.95, "val"),
    ("val-high", 2.40, "val"),
    ("test-low", 0.25, "test"),
    ("test-high", 5.10, "test"),
]


def save_archive(path, rows=SMALL_ROWS):
    cids, bandgaps, splits = zip(*rows)
    np.savez_compressed(
        path,
        cif_ids=np.asarray(cids),
        embeddings=np.zeros((len(rows), 4), dtype=np.float32),
        bandgaps=np.asarray(bandgaps, dtype=np.float64),
        splits=np.asarray(splits),
    )


def load_small_archive(path):
    return PAPER_SPLIT.load_and_validate_archive(
        path,
        expected_totals=SMALL_TOTALS,
        expected_positives=SMALL_POSITIVES,
    )


class PaperSplitArchiveTests(unittest.TestCase):
    def test_loads_valid_archive_and_preserves_archive_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "labeled.npz"
            save_archive(archive)

            split_data = load_small_archive(archive)

            self.assertEqual(
                list(split_data["train"]),
                ["train-low", "train-high-a", "train-high-b"],
            )
            self.assertEqual(list(split_data["val"]), ["val-low", "val-high"])
            self.assertEqual(list(split_data["test"]), ["test-low", "test-high"])

    def test_rejects_value_exactly_at_inclusive_boundary(self):
        rows = list(SMALL_ROWS)
        rows[1] = (rows[1][0], 1.0, rows[1][2])
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "boundary.npz"
            save_archive(archive, rows)

            with self.assertRaisesRegex(
                PAPER_SPLIT.SplitValidationError, "exactly at.*decision boundary"
            ):
                load_small_archive(archive)

    def test_rejects_duplicate_cif_id(self):
        rows = list(SMALL_ROWS)
        rows[-1] = (rows[0][0], rows[-1][1], rows[-1][2])
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "duplicate.npz"
            save_archive(archive, rows)

            with self.assertRaisesRegex(
                PAPER_SPLIT.SplitValidationError, "globally unique"
            ):
                load_small_archive(archive)


class NonOverwritingMaterializationTests(unittest.TestCase):
    def test_second_run_verifies_without_rewriting_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "labeled.npz"
            output_dir = tmp / "splits"
            save_archive(archive)
            split_data = load_small_archive(archive)

            first_states = PAPER_SPLIT.materialize_or_verify(split_data, output_dir)
            self.assertEqual(set(first_states.values()), {"created"})

            train_path = output_dir / "train_bandgaps_regression.json"
            compact_content = json.dumps(split_data["train"], separators=(",", ":"))
            train_path.write_text(compact_content, encoding="utf-8")

            second_states = PAPER_SPLIT.materialize_or_verify(split_data, output_dir)
            self.assertEqual(set(second_states.values()), {"verified"})
            self.assertEqual(train_path.read_text(encoding="utf-8"), compact_content)

    def test_mismatch_is_preserved_and_blocks_all_missing_creations(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "labeled.npz"
            output_dir = tmp / "splits"
            output_dir.mkdir()
            save_archive(archive)
            split_data = load_small_archive(archive)

            train_path = output_dir / "train_bandgaps_regression.json"
            stale_content = '{"stale-id": 0.2}\n'
            train_path.write_text(stale_content, encoding="utf-8")

            with self.assertRaisesRegex(
                PAPER_SPLIT.SplitValidationError, "refusing to overwrite"
            ):
                PAPER_SPLIT.materialize_or_verify(split_data, output_dir)

            self.assertEqual(train_path.read_text(encoding="utf-8"), stale_content)
            self.assertFalse((output_dir / "val_bandgaps_regression.json").exists())
            self.assertFalse((output_dir / "test_bandgaps_regression.json").exists())

    def test_verify_only_never_creates_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "labeled.npz"
            output_dir = tmp / "splits"
            output_dir.mkdir()
            save_archive(archive)
            split_data = load_small_archive(archive)

            with self.assertRaisesRegex(
                PAPER_SPLIT.SplitValidationError, "required split file is missing"
            ):
                PAPER_SPLIT.materialize_or_verify(
                    split_data, output_dir, verify_only=True
                )
            self.assertEqual(list(output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
