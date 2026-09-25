import unittest

from run_scripts.compare_representation_diagnostics import compare


def _row(texture2: str, baseline_distance: float, unigir_distance: float) -> dict:
    row = {
        "sample_index": 0,
        "base_index_1": 1,
        "base_index_2": 2,
        "texture1": "solid",
        "texture2": texture2,
        "full_aug_distance": 0.0,
        "masked_to_full_distance_a": baseline_distance,
        "masked_to_full_distance_b": baseline_distance,
        "masked_to_full_distance_mean": baseline_distance,
    }
    row["masked_to_full_distance_mean"] = unigir_distance
    return row


class RepresentationDiagnosticsCompareTest(unittest.TestCase):
    def test_compare_requires_paired_order_and_reports_delta(self):
        baseline_row = _row("pluses", 0.1, 0.1)
        unigir_row = _row("pluses", 0.2, 0.2)
        baseline = {"seed": 42, "profile_available": False, "rows": [baseline_row]}
        unigir = {"seed": 42, "profile_available": True, "rows": [unigir_row]}
        summary = compare(baseline, unigir)
        self.assertAlmostEqual(summary["overall"]["masked_to_full_distance_mean"]["delta"]["mean"], 0.1)
        self.assertEqual(summary["by_texture2"]["pluses"]["count"], 1)

    def test_compare_rejects_different_pair_order(self):
        baseline = {
            "seed": 42,
            "profile_available": False,
            "rows": [_row("solid", 0.1, 0.1)],
        }
        unigir_row = _row("pluses", 0.2, 0.2)
        unigir = {"seed": 42, "profile_available": True, "rows": [unigir_row]}
        with self.assertRaises(ValueError):
            compare(baseline, unigir)


if __name__ == "__main__":
    unittest.main()
