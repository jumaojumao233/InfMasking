from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
from evaluation.linear_probe import LinearProbingCallback

# pass the current epoch and total epochs to the model
class EpochInfoCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        pl_module.set_current_epoch(current_epoch)

class TotalEpochsCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        total_epochs = trainer.max_epochs
        pl_module.set_total_epochs(total_epochs)

@hydra.main(version_base=None, config_name="train_trifeatures", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on synthetic toy data (bimodal trifeatures) with
    controllable attributes (shape, color, texture).

    Models currently implemented are:
        - CLIP
        - CrossSelf
        - CoMM
        - InfMasking
    """

    # Seed Python, NumPy, Torch and DataLoader workers consistently.
    seed_everything(int(cfg.seed), workers=True)

    # create model + save hyper-parameters
    kwargs = dict()

    if cfg.model.name== "CoMM" or cfg.model.name == "InfMasking":
        kwargs["encoder"] = {
            "encoders": instantiate(cfg.model.encoders),
            "input_adapters": instantiate(cfg.model.adapters)}

    if cfg.model.name == "CLIP":
        encoders = instantiate(cfg.model.encoders)
        kwargs["visual"], kwargs["language"] = encoders[0], encoders[1]
        kwargs["image_projection"] = instantiate(cfg.model.clip_image_projection)
        kwargs["text_projection"] = instantiate(cfg.model.clip_text_projection)

    if cfg.model.name == "CrossSelf":
        encoders = instantiate(cfg.model.encoders)
        kwargs["enc1"] = encoders[0]
        kwargs["enc2"] = encoders[1]
        kwargs["head1"] = instantiate(cfg.model.visual_projection)
        kwargs["head2"] = instantiate(cfg.model.visual_projection)


    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)

    model.save_hyperparameters(cfg)

    # Data loading code
    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)

    # Linear probing on each task can be disabled for low-cost diagnostics.
    enable_linear_probe = getattr(cfg, "enable_linear_probe", True)
    configured_probe_names = getattr(cfg, "probe_names", None)
    downstream_names = list(configured_probe_names) if configured_probe_names else [
        "share", "unique1", "unique2", "synergy"
    ]
    downstream_data_modules = [instantiate(cfg.data.data_module, model="Sup", biased=False, task=t)
                               for t in downstream_names] if enable_linear_probe else []
    experiment_root = os.path.abspath(build_root_dir(cfg))
    checkpoint_dir = resolve_checkpoint_dir(cfg, experiment_root)
    logger_version = int(getattr(cfg, "logger_version", 0))
    logger = TensorBoardLogger(
        save_dir=experiment_root,
        name="logs",
        version=logger_version,
    )

    probe_frequency = getattr(cfg, "probe_frequency", "by_fit")
    checkpoint_monitor = getattr(cfg, "checkpoint_monitor", None)
    checkpoint_mode = getattr(cfg, "checkpoint_mode", "max")
    enable_early_stopping = getattr(cfg, "enable_early_stopping", False)
    if checkpoint_monitor is not None and probe_frequency != "by_epoch":
        raise ValueError("Metric-based checkpointing requires probe_frequency=by_epoch")
    if enable_early_stopping and checkpoint_monitor is None:
        raise ValueError("Early stopping requires checkpoint_monitor")

    checkpoint_callback = build_checkpoint_callback(
        checkpoint_dir,
        checkpoint_monitor=checkpoint_monitor,
        checkpoint_mode=checkpoint_mode,
    )

    early_stopping_callback = None
    if enable_early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor=checkpoint_monitor,
            patience=int(getattr(cfg, "early_stopping_patience", 10)),
            verbose=True,
            mode=checkpoint_mode,
        )
    # Trainer + fit
    callbacks = build_training_callbacks(
        cfg,
        downstream_data_modules,
        checkpoint_callback,
        probe_frequency,
        early_stopping_callback,
        downstream_names=downstream_names,
    )
    trainer = instantiate(
        cfg.trainer,
        default_root_dir=experiment_root,
        logger=logger,
        callbacks=callbacks,
    )

    if cfg.mode == "train":
        resume_ckpt_path = resolve_resume_ckpt_path(cfg, checkpoint_dir)
        print(f"Experiment root: {experiment_root}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Resume checkpoint: {resume_ckpt_path or 'none'}")
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=resume_ckpt_path,
        )
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=getattr(cfg, "ckpt_path", None))


def build_root_dir(cfg: DictConfig):
    # set directory for logs and checkpoints
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, "bimodal_trifeatures")

    # modify `root_dir` if in test mode to match pre-trained model's path
    if cfg.mode == "test":
        if cfg.ckpt_path is None:
            print(UserWarning("`ckpt_path` is not set during testing."))
        else:
            root_dir = os.path.join(os.path.dirname(cfg.ckpt_path), "test")

    if getattr(cfg, "exp_name", None) is not None:
        root_dir = os.path.join(root_dir, cfg.exp_name)

    return root_dir


def resolve_checkpoint_dir(cfg: DictConfig, experiment_root: str) -> str:
    """Return the stable checkpoint directory for one experiment."""
    configured_dir = getattr(cfg, "checkpoint_dir", None)
    if configured_dir in (None, ""):
        configured_dir = os.path.join(experiment_root, "checkpoints")
    return os.path.abspath(os.path.expanduser(str(configured_dir)))


def build_checkpoint_callback(
        checkpoint_dir: str,
        checkpoint_monitor=None,
        checkpoint_mode: str = "max") -> ModelCheckpoint:
    """Build a callback that always keeps the latest epoch checkpoint."""
    checkpoint_kwargs = dict(
        dirpath=checkpoint_dir,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_last=True,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
    )
    if checkpoint_monitor is None:
        checkpoint_kwargs.update(filename="{epoch:02d}", save_top_k=0)
    else:
        checkpoint_kwargs.update(
            filename="{epoch:02d}-{" + checkpoint_monitor + ":.4f}",
            auto_insert_metric_name=False,
            save_top_k=1,
        )
    return ModelCheckpoint(**checkpoint_kwargs)


def build_training_callbacks(
        cfg: DictConfig,
        downstream_data_modules,
        checkpoint_callback: ModelCheckpoint,
        probe_frequency: str,
        early_stopping_callback=None,
        downstream_names=None):
    """Keep checkpointing independent from optional linear probing."""
    if downstream_names is None:
        downstream_names = ["share", "unique1", "unique2", "synergy"]
    callbacks = [EpochInfoCallback(), TotalEpochsCallback(), checkpoint_callback]
    if getattr(cfg, "enable_linear_probe", True):
        callbacks.append(
            LinearProbingCallback(
                downstream_data_modules,
                names=downstream_names,
                val_loaders=False,
                frequency=probe_frequency,
                export_predictions_dir=getattr(cfg, "export_predictions_dir", None)
            )
        )
        if getattr(cfg, "enable_early_stopping", False):
            callbacks.append(early_stopping_callback)
    return callbacks


def resolve_resume_ckpt_path(cfg: DictConfig, checkpoint_dir: str):
    """Prefer an explicitly supplied checkpoint, otherwise resume from last.ckpt."""
    configured_path = getattr(cfg, "resume_ckpt_path", None)
    if configured_path not in (None, ""):
        return os.path.abspath(os.path.expanduser(str(configured_path)))

    last_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
    return last_checkpoint if os.path.isfile(last_checkpoint) else None


if __name__ == '__main__':
    main()
