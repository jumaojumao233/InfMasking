import unittest

from run_scripts.analyze_representation_diagnostics import summarize


class RepresentationDiagnosticsTest(unittest.TestCase):
    def test_summarize_groups_by_second_texture(self):
        rows = []
        for texture in ("solid", "pluses"):
            rows.append({
                "texture1": "solid",
                "texture2": texture,
                "full_aug_distance": 0.1 if texture == "solid" else 0.3,
                "full_aug_cosine": 0.9 if texture == "solid" else 0.7,
                "masked_to_full_distance_a": 0.2,
                "masked_to_full_distance_b": 0.4,
                "masked_to_full_distance_mean": 0.3,
                "profile_pred_accuracy": 1.0 if texture == "solid" else 0.0,
                "profile_kl": 0.1 if texture == "solid" else 0.5,
                "profile_target_confidence": 0.8,
                "profile_target_entropy": 0.2,
                "profile_prediction_confidence": 0.7,
            })
        summary = summarize({
            "method": "unigir",
            "seed": 42,
            "num_samples": 2,
            "profile_available": True,
            "fixed_eval_transform": "fixed",
            "rows": rows,
        })
        self.assertEqual(summary["by_texture2"]["solid"]["count"], 1)
        self.assertEqual(summary["by_texture2"]["pluses"]["count"], 1)
        self.assertAlmostEqual(summary["by_texture2"]["pluses"]["profile_kl"]["mean"], 0.5)

    def test_summarize_handles_baseline_without_profile_values(self):
        row = {
            "texture1": "solid",
            "texture2": "solid",
            "full_aug_distance": 0.1,
            "full_aug_cosine": 0.9,
            "masked_to_full_distance_a": 0.2,
            "masked_to_full_distance_b": 0.4,
            "masked_to_full_distance_mean": 0.3,
            "profile_pred_accuracy": None,
            "profile_kl": None,
            "profile_target_confidence": None,
            "profile_target_entropy": None,
            "profile_prediction_confidence": None,
        }
        summary = summarize({
            "method": "baseline",
            "seed": 42,
            "num_samples": 1,
            "profile_available": False,
            "fixed_eval_transform": "fixed",
            "rows": [row],
        })
        self.assertNotIn("profile_kl", summary["overall"])
        self.assertAlmostEqual(summary["overall"]["full_aug_distance"]["mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
