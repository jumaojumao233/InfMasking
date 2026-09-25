import unittest

import torch

from losses.geodesic_profile import GeodesicPrototypeAlignment
from losses.infmasking_loss import InfMaskingLoss


class GeodesicProfileTest(unittest.TestCase):
    def test_graph_profile_is_finite_and_reports_structure(self):
        torch.manual_seed(5)
        module = GeodesicPrototypeAlignment(
            dim=8,
            num_prototypes=8,
            queue_size=0,
            graph_k=3,
            graph_anchors=2,
            init_seed=5,
        )
        module.eval()
        z_a = torch.randn(4, 8)
        z_b = torch.randn(4, 8)
        masks_a = [torch.randn(4, 8), torch.randn(4, 8)]
        masks_b = [torch.randn(4, 8), torch.randn(4, 8)]

        output = module(z_a, masks_a, z_b, masks_b)

        for key in (
                "loss", "kl", "acc", "graph_num_components",
                "graph_unreachable_pairs", "graph_mean_degree",
                "graph_mean_edge_weight"):
            self.assertIn(key, output)
            self.assertTrue(torch.isfinite(output[key]))
        self.assertEqual(output["graph_num_components"].item(), 1)
        self.assertEqual(output["graph_unreachable_pairs"].item(), 0)

    def test_infmasking_loss_selects_geodesic_profile(self):
        loss_fn = InfMaskingLoss(
            temperature=0.1,
            mask_lambda=1.0,
            profile_kwargs={
                "dim": 4,
                "num_prototypes": 4,
                "queue_size": 0,
                "loss_weight": 0.25,
                "init_seed": 3,
                "distance_mode": "geodesic",
                "graph_k": 2,
                "graph_anchors": 2,
            },
        )

        self.assertIsInstance(loss_fn.profile, GeodesicPrototypeAlignment)

    def test_graph_refreshes_only_at_configured_epoch_boundary(self):
        module = GeodesicPrototypeAlignment(
            dim=4,
            num_prototypes=4,
            queue_size=0,
            graph_k=2,
            graph_update_frequency=2,
            init_seed=2,
        )
        module.eval()
        z = torch.randn(2, 4)
        masks = [torch.randn(2, 4)]

        module.set_epoch(0)
        module(z, masks, z, masks)
        first_graph = module.graph_distances.clone()
        module.set_epoch(1)
        module(z, masks, z, masks)
        self.assertTrue(torch.equal(first_graph, module.graph_distances))
        module.set_epoch(2)
        self.assertEqual(module.graph_distances.numel(), 0)


if __name__ == "__main__":
    unittest.main()
