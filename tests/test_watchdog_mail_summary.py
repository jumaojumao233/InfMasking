from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WatchdogMailSummaryTests(unittest.TestCase):
    def test_watchdogs_use_shared_summary_and_write_dryrun_body(self):
        formatter = ROOT / "run_scripts" / "winpc_watchdog_mail.ps1"
        formatter_text = formatter.read_text(encoding="utf-8")
        self.assertIn("Completed run names are omitted from this mail", formatter_text)
        self.assertIn("Current run(s):", formatter_text)
        self.assertIn("Failed or blocked run(s):", formatter_text)

        for name in ("winpc_experiment_watchdog.ps1", "winpc_g4_watchdog.ps1"):
            text = (ROOT / "run_scripts" / name).read_text(encoding="utf-8")
            self.assertIn("winpc_watchdog_mail.ps1", text)
            self.assertIn("DRY_RUN_BODY_WRITTEN", text)
            self.assertNotIn('"Run states:"', text)
            self.assertIn("$allowed = $desktopProcess -or $knownRun", text)

    def test_summary_does_not_reintroduce_full_completed_queue(self):
        formatter_text = (
            ROOT / "run_scripts" / "winpc_watchdog_mail.ps1"
        ).read_text(encoding="utf-8")
        self.assertNotIn("foreach ($run in $Result.Runs)", formatter_text)


if __name__ == "__main__":
    unittest.main()
