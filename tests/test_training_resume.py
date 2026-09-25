import tempfile
import unittest
from pathlib import Path
import json

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from main_trifeatures import (
    build_checkpoint_callback,
    build_training_callbacks,
    resolve_checkpoint_dir,
    resolve_resume_ckpt_path,
)
from evaluation.linear_probe import export_predictions


class TrainingResumeTest(unittest.TestCase):
    def test_resume_prefers_existing_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()
            last_checkpoint = checkpoint_dir / "last.ckpt"
            last_checkpoint.write_bytes(b"checkpoint")
            cfg = OmegaConf.create({"resume_ckpt_path": None})

            self.assertEqual(
                resolve_resume_ckpt_path(cfg, str(checkpoint_dir)),
                str(last_checkpoint),
            )

    def test_explicit_resume_path_has_priority(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = OmegaConf.create({"resume_ckpt_path": "manual.ckpt"})
            expected = str(Path.cwd() / "manual.ckpt")
            self.assertEqual(
                resolve_resume_ckpt_path(cfg, temp_dir),
                expected,
            )

    def test_missing_last_checkpoint_starts_from_scratch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = OmegaConf.create({"resume_ckpt_path": None})
            self.assertIsNone(resolve_resume_ckpt_path(cfg, temp_dir))

    def test_checkpoint_callback_always_saves_last(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = build_checkpoint_callback(temp_dir)
            self.assertIsInstance(callback, ModelCheckpoint)
            self.assertTrue(callback.save_last)
            self.assertEqual(callback.save_top_k, 0)
            self.assertEqual(Path(callback.dirpath), Path(temp_dir))

    def test_checkpoint_callback_is_kept_without_linear_probe(self):
        cfg = OmegaConf.create({
            "enable_linear_probe": False,
            "enable_early_stopping": False,
        })
        checkpoint_callback = build_checkpoint_callback(tempfile.mkdtemp())
        callbacks = build_training_callbacks(
            cfg,
            downstream_data_modules=[],
            checkpoint_callback=checkpoint_callback,
            probe_frequency="by_fit",
        )
        self.assertIn(checkpoint_callback, callbacks)
        self.assertEqual(len(callbacks), 3)

    def test_probe_predictions_are_exported_to_a_task_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            export_predictions(
                temp_dir,
                "unique2",
                {"y_true": [0, 1], "y_pred": [0, 0], "probabilities": [[0.9, 0.1], [0.6, 0.4]]},
            )
            output = Path(temp_dir) / "unique2.json"
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text())["y_pred"], [0, 0])


if __name__ == "__main__":
    unittest.main()
