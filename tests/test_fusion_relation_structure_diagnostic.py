import unittest

import numpy as np
import torch

from run_scripts.diagnose_fusion_relation_structure import (
    compute_metrics,
    relation_matrix,
    relation_upper_triangle,
    spearman_relation,
    topk_neighbor_retention,
)
from run_scripts.diagnose_interaction_signature_structure import interaction_signature
from run_scripts.diagnose_nonadditive_residual_structure import (
    concatenate_modalities,
    residual_from_prediction,
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

    def test_interaction_signature_contains_two_normalized_increments(self):
        full = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        mod1 = np.zeros_like(full)
        mod2 = np.ones_like(full)
        signature = interaction_signature(
            torch.from_numpy(full),
            torch.from_numpy(mod1),
            torch.from_numpy(mod2),
        ).numpy()
        self.assertEqual(signature.shape, (2, 4))
        self.assertTrue(np.allclose(np.linalg.norm(signature[:, :2], axis=1), 1.0))
        self.assertTrue(np.allclose(np.linalg.norm(signature[:, 2:], axis=1), 1.0))

    def test_residual_proxy_preserves_aligned_shapes(self):
        mod1 = np.ones((3, 2), dtype=np.float32)
        mod2 = np.full((3, 2), 2.0, dtype=np.float32)
        target = np.full((3, 2), 4.0, dtype=np.float32)
        prediction = np.full((3, 2), 3.0, dtype=np.float32)
        features = concatenate_modalities(mod1, mod2)
        residual = residual_from_prediction(target, prediction)
        self.assertEqual(features.shape, (3, 4))
        np.testing.assert_allclose(residual, 1.0)


if __name__ == "__main__":
    unittest.main()
