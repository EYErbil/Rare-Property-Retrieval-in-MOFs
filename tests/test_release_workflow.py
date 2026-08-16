import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPAIR_SCRIPT = REPO_ROOT / "screening" / "data_preparation" / "repair_split_symlinks.py"
REINFER_SCRIPT = REPO_ROOT / "screening" / "discovery" / "reinfer_ml.py"
ENSEMBLE_PREDICTIONS = REPO_ROOT / "screening" / "discovery" / "ensemble_predictions.py"
ENSEMBLE_DISCOVERY = REPO_ROOT / "screening" / "src" / "ensemble_discovery.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SplitLinkWorkflowTests(unittest.TestCase):
    def test_source_dir_cli_creates_all_required_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            splits = tmp / "splits"
            source = tmp / "source"
            splits.mkdir()
            (source / "test").mkdir(parents=True)
            (splits / "train_bandgaps_regression.json").write_text(
                json.dumps({"MOF_A": 0.5}), encoding="utf-8"
            )
            for extension in ("grid", "griddata16", "graphdata"):
                (source / "test" / f"MOF_A.{extension}").write_text(
                    extension, encoding="utf-8"
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPAIR_SCRIPT),
                    "--splits_dir",
                    str(splits),
                    "--source_dir",
                    str(source),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            for extension in ("grid", "griddata16", "graphdata"):
                target = splits / "train" / f"MOF_A.{extension}"
                self.assertTrue(target.is_symlink())
                self.assertTrue(target.exists())

    def test_missing_structure_source_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            splits = tmp / "splits"
            source = tmp / "source"
            splits.mkdir()
            source.mkdir()
            (splits / "train_bandgaps_regression.json").write_text(
                json.dumps({"MOF_A": 0.5}), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPAIR_SCRIPT),
                    "--splits_dir",
                    str(splits),
                    "--source_dir",
                    str(source),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)


class DiscoveryWorkflowTests(unittest.TestCase):
    def test_prospective_predictions_take_precedence(self):
        module = load_module("ensemble_predictions_release_test", ENSEMBLE_PREDICTIONS)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            test_csv = tmp / "test_predictions.csv"
            inference_csv = tmp / "inference_predictions.csv"
            test_csv.write_text("cif_id,score\nold,1\n", encoding="utf-8")
            inference_csv.write_text("cif_id,score\nnew,1\n", encoding="utf-8")
            self.assertEqual(module.find_predictions_csv(tmp), str(inference_csv))

    def test_requested_reinference_cannot_succeed_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "embeddings.npz"
            classifiers = tmp / "classifiers"
            classifiers.mkdir()
            np.savez_compressed(
                archive,
                cif_ids=np.asarray(["MOF_A"]),
                embeddings=np.zeros((1, 4), dtype=np.float32),
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(REINFER_SCRIPT),
                    "--embeddings_path",
                    str(archive),
                    "--clf_dir",
                    str(classifiers),
                    "--methods",
                    "smote_extra_trees",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(any(tmp.rglob("final_results.json")))

    def test_test_label_stacking_is_not_implemented(self):
        source = ENSEMBLE_DISCOVERY.read_text(encoding="utf-8")
        self.assertNotIn("def stacking_ensemble", source)
        self.assertNotIn("stack_scores", source)


if __name__ == "__main__":
    unittest.main()
