import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MosiProfileWeightAblationTests(unittest.TestCase):
    def test_mosi_runner_exposes_profile_weight(self):
        runner = (ROOT / "run_scripts" / "winpc_mosi_unigir_smoke.ps1").read_text(encoding="utf-8")
        scheduler = (ROOT / "run_scripts" / "winpc_schedule_mosi_experiment.ps1").read_text(encoding="utf-8")
        watchdog = (ROOT / "run_scripts" / "winpc_experiment_watchdog.ps1").read_text(encoding="utf-8")
        self.assertIn("$ProfileLossWeight = 0.25", runner)
        self.assertIn("profile_kwargs.loss_weight=$ProfileLossWeight", runner)
        self.assertIn("-ProfileLossWeight $ProfileLossWeight", scheduler)
        self.assertIn('SchedulerType -eq "mosi"', watchdog)
        self.assertIn('"-ProfileLossWeight", $profileLossWeight', watchdog)

    def test_ablation_config_is_single_variable_change(self):
        config_path = ROOT / "run_scripts" / "winpc_mosi_profile_weight_a0125.config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertFalse(config["AutoStart"])
        self.assertEqual(len(config["Runs"]), 1)
        run = config["Runs"][0]
        self.assertEqual(run["Method"], "unigir")
        self.assertEqual(run["Seed"], 42)
        self.assertEqual(run["MaxEpochs"], 10)
        self.assertEqual(run["BatchSize"], 32)
        self.assertEqual(run["ProfileLossWeight"], 0.125)
        self.assertEqual(run["ProbeFrequency"], "by_fit")

    def test_follow_up_queue_has_two_serial_runs(self):
        config_path = ROOT / "run_scripts" / "winpc_mosi_profile_weight_a0125_s7_s123.config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertTrue(config["AutoStart"])
        self.assertEqual(config["SchedulerType"], "mosi")
        self.assertEqual([run["Seed"] for run in config["Runs"]], [7, 123])
        self.assertEqual([run["ProfileLossWeight"] for run in config["Runs"]], [0.125, 0.125])


if __name__ == "__main__":
    unittest.main()
