import unittest

from run_scripts.aggregate_representation_diagnostics import aggregate


def _profile_summary(seed: int, value: float) -> dict:
    group = {
        "count": 1,
        "profile_pred_accuracy": {"mean": value},
        "profile_kl": {"mean": value + 0.1},
        "profile_target_confidence": {"mean": 0.5},
        "profile_target_entropy": {"mean": 0.2},
    }
    return {
        "seed": seed,
        "num_samples": 1,
        "by_texture2": {"solid": group, **{name: {"count": 0} for name in (
            "stripes", "grid", "hexgrid", "dots", "noise", "triangles", "zigzags", "rain", "pluses"
        )}},
    }


def _comparison(value: float) -> dict:
    group = {"count": 1, "masked_to_full_distance_mean": {"delta": {"mean": value}}}
    return {"by_texture2": {"solid": group, **{name: {"count": 0} for name in (
        "stripes", "grid", "hexgrid", "dots", "noise", "triangles", "zigzags", "rain", "pluses"
    )}}}


class AggregateRepresentationDiagnosticsTest(unittest.TestCase):
    def test_aggregate_keeps_seed_values_and_mean(self):
        result = aggregate(
            [_profile_summary(42, 0.7), _profile_summary(7, 0.8), _profile_summary(123, 0.9)],
            [_comparison(-0.1), _comparison(-0.2), _comparison(-0.3)],
        )
        group = result["by_texture2"]["solid"]
        self.assertEqual(group["counts"], [1, 1, 1])
        self.assertAlmostEqual(group["profile_pred_accuracy"]["mean"], 0.8)
        self.assertAlmostEqual(group["masked_to_full_distance_mean"]["mean"], -0.2)


if __name__ == "__main__":
    unittest.main()
