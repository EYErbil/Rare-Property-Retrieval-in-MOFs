import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / 'generation'
    / 'scripts'
    / 'materialize_generated_pool_manifest.py'
)


def load_module():
    spec = importlib.util.spec_from_file_location('generated_pool_manifest', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneratedPoolManifestTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_materializes_ids_in_descriptor_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / 'generated_soap_descriptors.npz'
            output = tmp / 'paper_generated_pool_manifest.txt'
            np.savez_compressed(
                archive,
                cif_ids=np.asarray(['pcu+N2+E3', 'hex+N1+E4', 'cds+N8+E9']),
                soap_descriptors=np.zeros((3, 2), dtype=np.float32),
            )

            count, digest = self.module.materialize_manifest(
                archive, output, expected_count=3
            )

            expected = b'pcu+N2+E3\nhex+N1+E4\ncds+N8+E9\n'
            self.assertEqual(count, 3)
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(digest, hashlib.sha256(expected).hexdigest())

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'duplicate.npz'
            np.savez_compressed(archive, cif_ids=np.asarray(['MOF_A', 'MOF_A']))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                self.module.load_identifiers(archive, expected_count=2)

    def test_rejects_wrong_paper_pool_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'short.npz'
            np.savez_compressed(archive, cif_ids=np.asarray(['MOF_A']))
            with self.assertRaisesRegex(ValueError, 'expected 13,802'):
                self.module.load_identifiers(archive)


if __name__ == '__main__':
    unittest.main()
