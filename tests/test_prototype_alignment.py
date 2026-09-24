import unittest

import torch
import torch.nn.functional as F

from losses.prototype_alignment import PrototypeAlignment


class PrototypeAlignmentTest(unittest.TestCase):
    def test_initialization_does_not_advance_global_rng(self):
        torch.manual_seed(123)
        state_before = torch.random.get_rng_state().clone()

        first = PrototypeAlignment(dim=8, num_prototypes=4, init_seed=7)

        self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))
        second = PrototypeAlignment(dim=8, num_prototypes=4, init_seed=7)
        self.assertTrue(torch.equal(first.prototypes, second.prototypes))

    def test_ema_update_uses_assignment_weighted_centroids(self):
        module = PrototypeAlignment(
            dim=2, num_prototypes=2, update_rate=0.25, init_seed=0)
        old = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
        module.prototypes.copy_(old)
        z = F.normalize(torch.tensor([[1.0, 1.0], [-1.0, 1.0]]), dim=1)
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        expected_centroids = z
        expected = F.normalize(0.75 * old + 0.25 * expected_centroids, dim=1)
        module._update(z, q)

        self.assertTrue(torch.allclose(module.prototypes, expected, atol=1e-6))

    def test_eval_forward_does_not_mutate_prototypes_or_queue(self):
        module = PrototypeAlignment(
            dim=4, num_prototypes=4, queue_size=8, init_seed=0)
        module.eval()
        z_a = F.normalize(torch.randn(3, 4), dim=1)
        z_b = F.normalize(torch.randn(3, 4), dim=1)
        masks_a = [F.normalize(torch.randn(3, 4), dim=1)]
        masks_b = [F.normalize(torch.randn(3, 4), dim=1)]
        prototypes_before = module.prototypes.clone()
        queue_before = module.feature_queue.clone()

        output = module(z_a, masks_a, z_b, masks_b)

        self.assertTrue(torch.equal(module.prototypes, prototypes_before))
        self.assertTrue(torch.equal(module.feature_queue, queue_before))
        self.assertEqual(int(module.queue_len.item()), 0)
        for key in (
                "loss", "acc", "kl", "usage_entropy", "active_prototypes",
                "target_entropy", "target_confidence",
                "prediction_usage_entropy", "prototype_mean_abs_cosine"):
            self.assertIn(key, output)
            self.assertTrue(torch.isfinite(output[key]))

    def test_training_forward_updates_queue_and_backpropagates(self):
        module = PrototypeAlignment(
            dim=4, num_prototypes=4, queue_size=8, init_seed=0)
        z_a = F.normalize(torch.randn(3, 4), dim=1)
        z_b = F.normalize(torch.randn(3, 4), dim=1)
        masks_a = [F.normalize(torch.randn(3, 4), dim=1).requires_grad_()]
        masks_b = [F.normalize(torch.randn(3, 4), dim=1).requires_grad_()]

        output = module(z_a, masks_a, z_b, masks_b)
        output["loss"].backward()

        self.assertEqual(int(module.queue_len.item()), 6)
        self.assertIsNotNone(masks_a[0].grad)
        self.assertIsNotNone(masks_b[0].grad)

    def test_shuffle_target_codes_is_deterministic_cyclic_shift(self):
        module = PrototypeAlignment(
            dim=4, num_prototypes=4, shuffle_targets=True, init_seed=0)
        q = torch.eye(4)

        shuffled = module._shuffle_target_codes(q)

        self.assertTrue(torch.equal(shuffled, q.roll(shifts=1, dims=0)))
        self.assertTrue(torch.equal(
            torch.sort(shuffled, dim=0).values,
            torch.sort(q, dim=0).values))
        self.assertTrue(torch.equal(
            module._shuffle_target_codes(q[:1]), q[:1]))

    def test_shuffle_only_changes_predictor_targets_not_state_updates(self):
        torch.manual_seed(19)
        normal = PrototypeAlignment(
            dim=4, num_prototypes=4, queue_size=8,
            shuffle_targets=False, init_seed=0)
        shuffled = PrototypeAlignment(
            dim=4, num_prototypes=4, queue_size=8,
            shuffle_targets=True, init_seed=0)
        z_a = F.normalize(torch.randn(3, 4), dim=1)
        z_b = F.normalize(torch.randn(3, 4), dim=1)
        masks_a = [F.normalize(torch.randn(3, 4), dim=1)]
        masks_b = [F.normalize(torch.randn(3, 4), dim=1)]

        normal_out = normal(z_a, masks_a, z_b, masks_b)
        shuffled_out = shuffled(z_a, masks_a, z_b, masks_b)

        self.assertTrue(torch.allclose(normal.prototypes, shuffled.prototypes))
        self.assertTrue(torch.equal(normal.feature_queue, shuffled.feature_queue))
        self.assertEqual(int(normal.queue_ptr.item()), int(shuffled.queue_ptr.item()))
        self.assertEqual(int(normal.queue_len.item()), int(shuffled.queue_len.item()))
        self.assertTrue(torch.isfinite(shuffled_out["loss"]))
        self.assertTrue(torch.isfinite(normal_out["loss"]))


if __name__ == "__main__":
    unittest.main()
