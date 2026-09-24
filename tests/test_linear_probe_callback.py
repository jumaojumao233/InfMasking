import unittest

from evaluation.linear_probe import LinearProbingCallback


class _Logger:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step):
        self.calls.append((metrics, step))


class _Trainer:
    def __init__(self):
        self.global_rank = 0
        self.global_step = 17
        self.logger = _Logger()


class LinearProbeCallbackTest(unittest.TestCase):
    def test_by_fit_skips_epoch_probe_and_logs_final_metrics(self):
        callback = LinearProbingCallback([], names=[], frequency="by_fit")
        trainer = _Trainer()
        calls = []

        def fake_probe(_trainer, _module, log=True):
            calls.append(log)
            return {"roc_auc_synergy": 0.61, "acc1": 0.37}

        callback.val_linear_probing = fake_probe
        callback.on_validation_epoch_end(trainer, object())
        self.assertEqual(calls, [])

        callback.on_fit_end(trainer, object())
        self.assertEqual(calls, [False])
        self.assertEqual(trainer.logger.calls, [({
            "final_roc_auc_synergy": 0.61,
            "final_acc1": 0.37,
        }, 17)])


if __name__ == "__main__":
    unittest.main()
