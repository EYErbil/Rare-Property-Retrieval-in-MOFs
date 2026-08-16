import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "generation" / "nominate_diverse_dft.py"
SPEC = importlib.util.spec_from_file_location("nominate_diverse_dft", SCRIPT_PATH)
NOMINATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NOMINATE)


def save_custom_csr(path, cids, matrix):
    matrix = sparse.csr_matrix(matrix)
    np.savez_compressed(
        path,
        cif_ids=np.asarray(cids),
        sp_data=matrix.data,
        sp_indices=matrix.indices,
        sp_indptr=matrix.indptr,
        sp_shape=np.asarray(matrix.shape),
    )


def save_predictions(path, cids, offset=0.0, reverse=False):
    scores = list(range(len(cids)))
    if reverse:
        scores.reverse()
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cif_id", "score", "mode"])
        for cid, score in zip(cids, scores):
            writer.writerow([cid, score + offset, "regression"])


class DiversityMatrixLoadingTests(unittest.TestCase):
    def test_dense_and_custom_csr_loaders_preserve_values(self):
        cids = ["a", "b", "c", "d"]
        values = np.asarray(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0],
             [4.0, 0.0, 5.0], [0.0, 6.0, 7.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dense_path = tmp / "dense.npz"
            sparse_path = tmp / "sparse.npz"
            np.savez_compressed(dense_path, cif_ids=np.asarray(cids), embeddings=values)
            save_custom_csr(sparse_path, cids, values)

            dense_ids, dense_matrix, dense_key, dense_format = (
                NOMINATE.load_diversity_matrix(dense_path)
            )
            sparse_ids, sparse_matrix, sparse_key, sparse_format = (
                NOMINATE.load_diversity_matrix(sparse_path)
            )

            self.assertEqual(dense_ids, cids)
            self.assertEqual(sparse_ids, cids)
            self.assertEqual(dense_key, "embeddings")
            self.assertEqual(dense_format, "dense")
            self.assertEqual(sparse_key, "sp_*")
            self.assertEqual(sparse_format, "custom CSR")
            self.assertTrue(sparse.isspmatrix_csr(sparse_matrix))
            np.testing.assert_allclose(dense_matrix, sparse_matrix.toarray())

            sparse_subset = NOMINATE.take_matrix_rows(sparse_matrix, [3, 1])
            self.assertTrue(sparse.isspmatrix_csr(sparse_subset))
            np.testing.assert_allclose(sparse_subset.toarray(), values[[3, 1]])

    def test_prediction_universe_requires_every_model(self):
        cids = ["labelled", "u0", "u1", "u2", "m1_only"]
        models = {
            "m1": {cid: float(i) for i, cid in enumerate(cids)},
            "m2": {"u0": 0.0, "u1": 1.0, "u2": 2.0},
        }
        self.assertEqual(
            NOMINATE.common_prediction_universe(cids, models),
            ["u0", "u1", "u2"],
        )

    def test_clustering_accepts_dense_and_sparse_matrices(self):
        values = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.9, 0.1]],
            dtype=np.float32,
        )
        for matrix in (values, sparse.csr_matrix(values)):
            with self.subTest(storage=type(matrix).__name__):
                labels, silhouette = NOMINATE.cluster_pool(
                    matrix, n_clusters=3, seed=7, pca_components=3,
                    kmeans_n_init=2,
                )
                self.assertEqual(labels.shape, (len(values),))
                self.assertTrue(np.isfinite(silhouette))


class NominationUniverseIntegrationTests(unittest.TestCase):
    def test_full_cache_rows_missing_from_one_prediction_never_leak(self):
        cids = ["labelled", "u0", "u1", "u2", "u3", "u4", "u5", "m1_only"]
        common_cids = ["u0", "u1", "u2", "u3", "u4", "u5"]
        values = np.eye(len(cids), 12, dtype=np.float32)
        results_by_storage = {}

        for storage in ("dense", "sparse"):
            with self.subTest(storage=storage), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                matrix_path = tmp / f"soap_{storage}.npz"
                if storage == "dense":
                    np.savez_compressed(
                        matrix_path, cif_ids=np.asarray(cids), embeddings=values
                    )
                else:
                    save_custom_csr(matrix_path, cids, values)

                model_1 = tmp / "model_1.csv"
                model_2 = tmp / "model_2.csv"
                save_predictions(model_1, cids)
                save_predictions(model_2, common_cids, offset=0.25, reverse=True)
                output_dir = tmp / "output"
                argv = [
                    str(SCRIPT_PATH),
                    "--embeddings_path", str(matrix_path),
                    "--prediction_csvs",
                    f"m1={model_1}", f"m2={model_2}",
                    "--nn_models", "m1",
                    "--ml_models", "m2",
                    "--output_dir", str(output_dir),
                    "--pool_size", "3",
                    "--pca_components", "2",
                    "--n_clusters", "2",
                    "--kmeans_n_init", "2",
                    "--budget", "2",
                    "--mmr_lambdas", "0.5",
                    "--exploration_budget", "1",
                    "--exploration_pool_hi", "6",
                ]

                with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    NOMINATE, "plot_umap", return_value=None
                ):
                    self.assertEqual(NOMINATE.main(), 0)

                with open(output_dir / "shortlist_pool.csv", encoding="utf-8") as handle:
                    shortlist_rows = list(csv.DictReader(handle))
                shortlist = {row["cif_id"] for row in shortlist_rows}
                nominee_order = (
                    (output_dir / "FINAL_TOP25_diverse.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                nominees = set(nominee_order)
                with open(
                    output_dir / "FINAL_TOP25_diverse.csv", encoding="utf-8"
                ) as handle:
                    nominee_rows = list(csv.DictReader(handle))
                self.assertLessEqual(shortlist, set(common_cids))
                self.assertLessEqual(nominees, set(common_cids))
                self.assertNotIn("labelled", shortlist | nominees)
                self.assertNotIn("m1_only", shortlist | nominees)
                self.assertTrue(
                    any(float(row["nn_ml_disagreement"]) > 0 for row in nominee_rows)
                )
                results_by_storage[storage] = (shortlist_rows, nominee_order)

        self.assertEqual(results_by_storage["dense"], results_by_storage["sparse"])

    def test_canonical_budget_reserves_five_soap_exploration_slots(self):
        rng = np.random.default_rng(42)
        cids = [f"mof_{index:04d}" for index in range(600)]
        descriptors = rng.normal(size=(len(cids), 16)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            matrix_path = tmp / "soap.npz"
            np.savez_compressed(
                matrix_path,
                cif_ids=np.asarray(cids),
                soap_descriptors=descriptors,
            )
            nn_csv = tmp / "nn.csv"
            ml_csv = tmp / "ml.csv"
            save_predictions(nn_csv, cids)
            save_predictions(ml_csv, cids, reverse=True)
            output_dir = tmp / "output"
            argv = [
                str(SCRIPT_PATH),
                "--embeddings_path", str(matrix_path),
                "--embedding_key", "soap_descriptors",
                "--embedding_label", "SOAP",
                "--prediction_csvs", f"nn={nn_csv}", f"ml={ml_csv}",
                "--nn_models", "nn",
                "--ml_models", "ml",
                "--output_dir", str(output_dir),
                "--pool_size", "500",
                "--pca_components", "50",
                "--n_clusters", "20",
                "--kmeans_n_init", "10",
                "--max_per_cluster", "1",
                "--mmr_lambdas", "0.2", "0.3", "0.4",
                "--alpha", "0.5",
                "--beta", "0.3",
                "--gamma", "0.2",
                "--budget", "25",
                "--exploration_budget", "5",
                "--exploration_pool_lo", "500",
                "--exploration_pool_hi", "2000",
                "--exploration_disagreement_weight", "0.6",
                "--exploration_rank_std_weight", "0.4",
                "--exploration_mmr_lambda", "0.4",
                "--rrf_k", "60",
                "--seed", "42",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                NOMINATE, "plot_umap", return_value=None
            ):
                self.assertEqual(NOMINATE.main(), 0)

            with open(
                output_dir / "FINAL_TOP25_diverse.csv", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 25)
            self.assertTrue(all(int(row["cluster"]) >= 0 for row in rows[:20]))
            self.assertTrue(all(int(row["cluster"]) == -1 for row in rows[20:]))
            self.assertTrue(all(int(row["rrf_rank"]) > 500 for row in rows[20:]))


if __name__ == "__main__":
    unittest.main()
