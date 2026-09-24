from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import numpy as np
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback,ModelCheckpoint,EarlyStopping
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

# pass the current epoch and total epochs to the model
class EpochInfoCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        pl_module.set_current_epoch(current_epoch)

class TotalEpochsCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        total_epochs = trainer.max_epochs
        pl_module.set_total_epochs(total_epochs)


@hydra.main(version_base=None, config_name="train_multibench", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on MultiBench dataset.
    Models currently implemented are:
        - CLIP
        - CrossSelf
        - CoMM
        - InfMasking
    """

    # Fix Python, NumPy, Torch and DataLoader worker seeds consistently.
    seed_everything(int(cfg.seed), workers=True)

    # create model + save hyper-parameters
    dataset = cfg.data.data_module.dataset # Which MultiBench dataset to load
    kwargs = dict()
    if cfg.model.name  == "CoMM" or cfg.model.name == "InfMasking":
        encoders = instantiate(cfg[dataset]["encoders"]) # encoders specific to each dataset
        adapters = instantiate(cfg[dataset]["adapters"]) # adapters also specific
        kwargs["encoder"] = {
            "encoders": encoders,
            "input_adapters": adapters}
    elif cfg.model.name == "CLIP":
        encoders = instantiate(cfg[dataset]["encoders"]) # encoders specific to each dataset
        kwargs["visual"], kwargs["language"] = encoders[0], encoders[1]
        kwargs["image_projection"] = instantiate(cfg[dataset].clip_projection1)
        kwargs["text_projection"] = instantiate(cfg[dataset].clip_projection2)
    elif cfg.model.name == "CrossSelf":
        encoders = instantiate(cfg[dataset]["encoders"])
        kwargs["enc1"] = encoders[0]
        kwargs["enc2"] = encoders[1]
        kwargs["head1"] = instantiate(cfg[dataset].projection_head1)
        kwargs["head2"] = instantiate(cfg[dataset].projection_head2)

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)

    model.save_hyperparameters(cfg)

    # Data loading code
    data_module = instantiate(cfg.data.data_module,
                              model=cfg.model.name,
                              modalities=cfg[dataset]["modalities"],
                              task=cfg[dataset]["task"],
                              **cfg[dataset]["kwargs"])

    downstream_data_module = instantiate(cfg.data.data_module,
                                         model="Sup",
                                         modalities=cfg[dataset]["modalities"],
                                         task=cfg[dataset]["task"])

    experiment_root = os.path.abspath(build_root_dir(cfg))
    checkpoint_dir = resolve_checkpoint_dir(cfg, experiment_root)
    logger_version = int(getattr(cfg, "logger_version", 0))
    logger = TensorBoardLogger(
        save_dir=experiment_root,
        name="logs",
        version=logger_version,
    )

    checkpoint_callback = build_checkpoint_callback(
        checkpoint_dir,
        checkpoint_monitor=getattr(cfg, "checkpoint_monitor", None),
        checkpoint_mode=getattr(cfg, "checkpoint_mode", "max"),
    )
    
    # Trainer + fit
    callbacks = build_training_callbacks(
        cfg,
        downstream_data_module,
        dataset,
        checkpoint_callback,
    )
    trainer = instantiate(
        cfg.trainer,
        default_root_dir=experiment_root,
        logger=[logger],
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
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, cfg.data.data_module.dataset)
    # modify `root_dir` if in test mode to match pre-trained model's path
    if cfg.mode == "test":
        ckpt_path = getattr(cfg, "ckpt_path", None)
        if ckpt_path is None:
            print(UserWarning("`ckpt_path` is not set during testing."))
        else:
            root_dir = os.path.join(os.path.dirname(ckpt_path), "test")

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
    """Save the latest epoch independently of optional probing metrics."""
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
        downstream_data_module,
        dataset: str,
        checkpoint_callback: ModelCheckpoint):
    """Keep checkpointing independent from optional linear probing."""
    callbacks = [EpochInfoCallback(), TotalEpochsCallback(), checkpoint_callback]
    if getattr(cfg, "enable_linear_probe", True):
        callbacks.append(
            instantiate(
                cfg.linear_probing,
                downstream_data_modules=[downstream_data_module],
                names=[dataset],
            )
        )
    return callbacks


def resolve_resume_ckpt_path(cfg: DictConfig, checkpoint_dir: str):
    """Prefer an explicit checkpoint, otherwise resume from this experiment's last.ckpt."""
    configured_path = getattr(cfg, "resume_ckpt_path", None)
    if configured_path not in (None, ""):
        return os.path.abspath(os.path.expanduser(str(configured_path)))

    last_checkpoint = os.path.join(checkpoint_dir, "last.ckpt")
    return last_checkpoint if os.path.isfile(last_checkpoint) else None


if __name__ == '__main__':

    main()
