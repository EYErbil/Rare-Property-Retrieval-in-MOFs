"""Regression tests for neutral-parent linker identity handling."""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
NOVELTY_PATH = REPO / "generation/scripts/novelty/chemical_identity_novelty.py"
IDENTIFIER_PATH = REPO / "generation/scripts/resolve_linker_smiles.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(shutil.which("obabel"), "Open Babel is required")
class NeutralParentRegressionTests(unittest.TestCase):
    def test_e146_charged_and_neutral_forms_collapse_to_one_parent_key(self):
        novelty = _load(NOVELTY_PATH, "novelty_e146")
        keys = novelty.obabel_skeletons(
            [novelty.E146_CHARGED_SMILES, novelty.E146_NEUTRAL_SMILES]
        )
        self.assertEqual(keys[novelty.E146_CHARGED_SMILES], "LBSASQXIHJDQCN")
        self.assertEqual(keys[novelty.E146_NEUTRAL_SMILES], "LBSASQXIHJDQCN")

    def test_every_hit_and_control_query_uses_normalization_path(self):
        novelty = _load(NOVELTY_PATH, "novelty_queries")
        hits, controls = novelty.normalized_queries()
        self.assertEqual(len(hits), 6)
        self.assertEqual(len(controls), 4)
        e146 = next(h for h in hits if h[0] == "qtz+N307+E146")
        self.assertEqual(e146[3], "LBSASQXIHJDQCN")

    def test_qmof_loader_ignores_raw_charged_mofkey_for_matching(self):
        novelty = _load(NOVELTY_PATH, "novelty_qmof")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "qmof.csv"
            fields = [
                "info.mofid.mofkey",
                "info.mofid.smiles_linkers",
                "info.mofid.smiles_nodes",
                "info.mofid.topology",
            ]
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "info.mofid.mofkey": "Cu.GTIBXBZXOUEIQT.MOFkey-v1.qtz",
                    "info.mofid.smiles_linkers": repr([novelty.E146_CHARGED_SMILES]),
                    "info.mofid.smiles_nodes": repr(["[Cu]"]),
                    "info.mofid.topology": "qtz",
                })
            rows = []
            novelty.load_qmof(path, rows)
        self.assertEqual(rows, [("QMOF", {"Cu"}, {"LBSASQXIHJDQCN"}, "qtz")])

    def test_mofdb_loader_neutralizes_mofid_linker_smiles(self):
        novelty = _load(NOVELTY_PATH, "novelty_mofdb")
        record = {
            "mofid": f"[Cu].{novelty.E146_CHARGED_SMILES} MOFid-v1.qtz.cat0",
            "mofkey": "Cu.GTIBXBZXOUEIQT.MOFkey-v1.qtz",
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "one.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            rows = []
            novelty.load_mofdb_json("testdb", td, rows)
        self.assertEqual(rows, [("testdb", {"Cu"}, {"LBSASQXIHJDQCN"}, "qtz")])

    def test_identifier_neutralizes_e146_but_retains_e151_stereo(self):
        identifiers = _load(IDENTIFIER_PATH, "linker_identifiers")
        e146 = identifiers.analyze_edge("E146")
        e151 = identifiers.analyze_edge("E151")

        self.assertEqual(e146["inchikey_linker"], "LBSASQXIHJDQCN-UHFFFAOYSA-N")
        self.assertEqual(e146["ob_formula_linker"], "C5H2N4")
        self.assertIn("one-proton parent", e146["identifier_note"])
        self.assertEqual(e151["inchikey_linker"], "PXGZQGDTEZPERC-IZLXSQMJSA-N")
        self.assertIn("@", e151["smiles_linker"])


if __name__ == "__main__":
    unittest.main()
