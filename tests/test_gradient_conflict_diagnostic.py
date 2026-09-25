import unittest

import torch

from losses.infmasking_loss import InfMaskingLoss
from run_scripts.gradient_conflict_diagnostic import gradient_metrics, summarize_records


class GradientConflictDiagnosticTest(unittest.TestCase):
    def test_opposite_gradients_are_marked_as_conflict(self):
        result = gradient_metrics(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))

        self.assertAlmostEqual(result["cosine"], -1.0, places=6)
        self.assertTrue(result["conflict"])

    def test_component_losses_are_available_on_request(self):
        torch.manual_seed(3)
        loss_fn = InfMaskingLoss(
            temperature=0.1,
            mask_lambda=1.0,
            profile_kwargs={
                "dim": 4,
                "num_prototypes": 4,
                "queue_size": 0,
                "loss_weight": 0.25,
                "init_seed": 3,
            },
        )
        z1 = torch.randn(3, 4, requires_grad=True)
        z2 = torch.randn(3, 4, requires_grad=True)
        mask1 = torch.randn(3, 4, requires_grad=True)
        mask2 = torch.randn(3, 4, requires_grad=True)
        outputs = {
            "aug1_embed": [z1],
            "aug2_embed": [z2],
            "mask_out1": [[mask1]],
            "mask_out2": [[mask2]],
            "prototype": -1,
            "epoch": 9,
            "max_epochs": 10,
        }

        result = loss_fn(outputs, return_components=True)

        self.assertIn("loss_base_raw", result)
        self.assertIn("loss_profile_raw", result)
        self.assertTrue(result["loss_base_raw"].requires_grad)
        self.assertTrue(result["loss_profile_raw"].requires_grad)
        self.assertTrue(torch.allclose(
            result["loss"], result["loss_base_raw"] + result["loss_profile_raw"]))

    def test_summary_counts_conflicting_batches(self):
        records = [
            {
                "all": {
                    "base_norm": 1.0,
                    "profile_norm": 1.0,
                    "profile_to_base_norm_ratio": 1.0,
                    "dot_product": -1.0,
                    "cosine": -1.0,
                    "conflict": True,
                },
                "encoder": {
                    "base_norm": 1.0,
                    "profile_norm": 1.0,
                    "profile_to_base_norm_ratio": 1.0,
                    "dot_product": -1.0,
                    "cosine": -1.0,
                    "conflict": True,
                },
                "head": {
                    "base_norm": 1.0,
                    "profile_norm": 1.0,
                    "profile_to_base_norm_ratio": 1.0,
                    "dot_product": -1.0,
                    "cosine": -1.0,
                    "conflict": True,
                },
            },
            {
                "all": {
                    "base_norm": 1.0,
                    "profile_norm": 1.0,
                    "profile_to_base_norm_ratio": 1.0,
                    "dot_product": 1.0,
                    "cosine": 1.0,
                    "conflict": False,
                },
                "encoder": {
                    "base_norm": 1.0,
                    "profile_norm": 1.0,
                    "profile_to_base_norm_ratio": 1.0,
                    "dot_product": 1.0,
                    "cosine": 1.0,
                    "conflict": False,
                },
                "head": {
                    "base_norm": 1.0,
                    "profile_norm": 1.0,
                    "profile_to_base_norm_ratio": 1.0,
                    "dot_product": 1.0,
                    "cosine": 1.0,
                    "conflict": False,
                },
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["all"]["conflict_count"], 1)
        self.assertEqual(summary["all"]["conflict_fraction"], 0.5)
        self.assertAlmostEqual(summary["all"]["metrics"]["cosine"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
