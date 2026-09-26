import unittest

import numpy as np

from run_scripts.diagnose_fusion_relation_structure import (
    compute_metrics,
    relation_matrix,
    relation_upper_triangle,
    spearman_relation,
    topk_neighbor_retention,
)


class FusionRelationStructureDiagnosticTest(unittest.TestCase):
    def test_relation_matrix_and_upper_triangle(self):
        embeddings = np.eye(3, dtype=np.float32)
        relation = relation_matrix(embeddings)
        np.testing.assert_allclose(np.diag(relation), 1.0)
        np.testing.assert_allclose(relation_upper_triangle(relation), 0.0)

    def test_identical_relations_are_preserved(self):
        embeddings = np.random.default_rng(7).normal(size=(24, 8)).astype(np.float32)
        metrics = compute_metrics(embeddings, embeddings.copy())
        self.assertAlmostEqual(metrics["spearman_relation"], 1.0)
        self.assertAlmostEqual(metrics["self_cosine"], 1.0)
        self.assertAlmostEqual(metrics["top5_retention"], 1.0)

    def test_neighbor_retention_detects_reordered_structure(self):
        embeddings = np.eye(6, dtype=np.float32)
        full = relation_matrix(embeddings)
        reordered = embeddings[[1, 0, 2, 3, 4, 5]]
        masked = relation_matrix(reordered)
        self.assertAlmostEqual(topk_neighbor_retention(full, masked, 5), 1.0)
        self.assertAlmostEqual(spearman_relation(full, masked), 1.0)

    def test_invalid_top_k_is_rejected(self):
        relation = np.eye(4, dtype=np.float32)
        with self.assertRaises(ValueError):
            topk_neighbor_retention(relation, relation, 4)


if __name__ == "__main__":
    unittest.main()
