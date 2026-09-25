import unittest

from run_scripts.analyze_profile_diagnostics import parse_test_results


class ProfileDiagnosticsTest(unittest.TestCase):
    def test_parse_test_results_extracts_numeric_profile_metrics(self):
        text = "Test results: [{'test_profile_kl': 0.125, 'test_profile_target_confidence': 0.42, 'test_note': 'skip'}]"
        self.assertEqual(
            parse_test_results(text),
            {"test_profile_kl": 0.125, "test_profile_target_confidence": 0.42},
        )

    def test_parse_test_results_requires_aggregated_output(self):
        with self.assertRaises(ValueError):
            parse_test_results("no aggregate output")


if __name__ == "__main__":
    unittest.main()
