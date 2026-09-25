import tempfile
import unittest
from pathlib import Path
import json

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from main_trifeatures import (
    build_root_dir,
    build_checkpoint_callback,
    build_training_callbacks,
    resolve_checkpoint_dir,
    resolve_resume_ckpt_path,
)
from dataset.trifeatures import build_trifeatures_image_transform
from evaluation.linear_probe import export_predictions
from torchvision import transforms


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

    def test_fixed_eval_transform_uses_resize_and_center_crop(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        fixed = build_trifeatures_image_transform(normalize, fixed_eval_transform=True)
        random = build_trifeatures_image_transform(normalize, fixed_eval_transform=False)

        self.assertIsInstance(fixed.transforms[0], transforms.Resize)
        self.assertIsInstance(fixed.transforms[1], transforms.CenterCrop)
        self.assertIsInstance(random.transforms[0], transforms.RandomResizedCrop)

    def test_test_root_dir_overrides_checkpoint_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = OmegaConf.create({
                "mode": "test",
                "ckpt_path": "C:/checkpoint/last.ckpt",
                "test_root_dir": temp_dir,
                "exp_name": "fixed_eval",
                "trainer": {"default_root_dir": "C:/default"},
                "model": {"name": "InfMasking"},
            })
            self.assertEqual(
                build_root_dir(cfg),
                str(Path(temp_dir) / "fixed_eval"),
            )


if __name__ == "__main__":
    unittest.main()
