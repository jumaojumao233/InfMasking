"""Measure local gradient conflict between InfMasking and profile losses.

The diagnostic loads one trained UniGIR checkpoint, runs a small number of
training-data batches in eval mode, and never calls ``optimizer.step``.  It
compares gradients of the original InfMasking loss with the *weighted*
profile loss on the encoder and projection-head parameters.  Prototype and
queue buffers are checked before and after the run so the command is strictly
read-only with respect to the checkpoint and model state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional
from pathlib import Path
from statistics import mean, stdev

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_config(project_root: Path, seed: int, data_root: Path,
                 batch_size: int, max_size: int, fixed_eval_transform: bool):
    """Build the same 256-dim UniGIR configuration used by G0."""
    base = OmegaConf.load(project_root / "configs" / "train_trifeatures.yaml")
    model_group = OmegaConf.load(project_root / "configs" / "model" / "unigir.yaml")
    data_group = OmegaConf.load(project_root / "configs" / "data" / "trifeatures.yaml")
    merged_model = OmegaConf.merge(base.model, model_group)
    cfg = OmegaConf.merge(base, {"model": merged_model, "data": data_group})

    cfg.seed = int(seed)
    cfg.mode = "train"
    cfg.data.data_module.data_root = str(data_root)
    cfg.data.data_module.batch_size = int(batch_size)
    cfg.data.data_module.num_workers = 0
    cfg.data.data_module.fixed_eval_transform = bool(fixed_eval_transform)
    cfg.data.data_module.max_size = int(max_size)
    cfg.data.data_module.biased = True
    cfg.data.data_module.seed = int(seed)

    # Match the repaired G0 architecture explicitly instead of relying on a
    # checkpoint's serialized hparams to mutate the data protocol.
    cfg.model.model.encoder.embed_dim = 256
    cfg.model.adapters[0].dim_tokens = 256
    cfg.model.adapters[1].dim_tokens = 256
    cfg.model.model.encoder.num_mask = 3
    cfg.model.model.encoder.mask_ratio = 0.7
    cfg.model.model.loss_kwargs.profile_kwargs.num_prototypes = 128
    cfg.model.model.loss_kwargs.profile_kwargs.queue_size = 1024
    cfg.model.model.loss_kwargs.profile_kwargs.loss_weight = 0.25
    cfg.model.model.loss_kwargs.profile_kwargs.cross = False
    cfg.model.model.loss_kwargs.profile_kwargs.shuffle_targets = False
    OmegaConf.resolve(cfg)
    return cfg


def build_model(cfg):
    kwargs = {
        "encoder": {
            "encoders": instantiate(cfg.model.encoders),
            "input_adapters": instantiate(cfg.model.adapters),
        }
    }
    return instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not state_dict:
        raise ValueError(f"checkpoint does not contain a state_dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint state mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def move_inputs(batch, device: torch.device):
    """Move the two modality lists returned by TrifeaturesMMSSL."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("expected a pair of modality batches")
    moved = []
    for view in batch:
        if not isinstance(view, (tuple, list)):
            raise ValueError("each view must be a list of modality tensors")
        moved.append([value.to(device, non_blocking=True) for value in view])
    return tuple(moved)


def _flatten_grads(parameters, gradients, indices=None) -> torch.Tensor:
    selected = indices if indices is not None else range(len(parameters))
    pieces = []
    for index in selected:
        parameter = parameters[index]
        gradient = gradients[index]
        if gradient is None:
            pieces.append(torch.zeros_like(parameter, memory_format=torch.contiguous_format).reshape(-1))
        else:
            pieces.append(gradient.detach().reshape(-1))
    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces)


def gradient_metrics(base_gradient: torch.Tensor,
                     profile_gradient: torch.Tensor) -> dict[str, float | bool | None]:
    """Compute direction and magnitude statistics for two flattened gradients."""
    base_norm = float(torch.linalg.vector_norm(base_gradient).item())
    profile_norm = float(torch.linalg.vector_norm(profile_gradient).item())
    dot = float(torch.dot(base_gradient, profile_gradient).item())
    denominator = base_norm * profile_norm
    cosine = dot / denominator if denominator > 0.0 else None
    ratio = profile_norm / base_norm if base_norm > 0.0 else None
    return {
        "base_norm": base_norm,
        "profile_norm": profile_norm,
        "profile_to_base_norm_ratio": ratio,
        "dot_product": dot,
        "cosine": cosine,
        "conflict": bool(cosine is not None and cosine < 0.0),
    }


def _finite_mean(values: list[float]) -> Optional[float]:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else None


def summarize_records(records: list[dict]) -> dict:
    metric_names = (
        "base_norm",
        "profile_norm",
        "profile_to_base_norm_ratio",
        "dot_product",
        "cosine",
    )
    aggregate = {}
    for group in ("all", "encoder", "head"):
        group_records = [row[group] for row in records]
        values = {}
        for name in metric_names:
            samples = [
                float(row[name]) for row in group_records
                if row[name] is not None and math.isfinite(float(row[name]))
            ]
            values[name] = {
                "mean": _finite_mean(samples),
                "std": stdev(samples) if len(samples) > 1 else 0.0,
                "min": min(samples, default=None),
                "max": max(samples, default=None),
            }
        conflict_count = sum(bool(row["conflict"]) for row in group_records)
        aggregate[group] = {
            "num_batches": len(group_records),
            "conflict_count": conflict_count,
            "conflict_fraction": conflict_count / len(group_records) if group_records else None,
            "metrics": values,
        }
    return aggregate


def _parameter_groups(named_parameters):
    parameters = []
    groups = {"encoder": [], "head": [], "all": []}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        index = len(parameters)
        parameters.append(parameter)
        groups["all"].append(index)
        if name.startswith("encoder."):
            groups["encoder"].append(index)
        elif name.startswith("head."):
            groups["head"].append(index)
    if not parameters:
        raise ValueError("no trainable model parameters were found")
    return parameters, groups


def _state_snapshot(profile):
    snapshot = {"prototypes": profile.prototypes.detach().clone()}
    for name in ("feature_queue", "queue_ptr", "queue_len"):
        if hasattr(profile, name):
            snapshot[name] = getattr(profile, name).detach().clone()
    return snapshot


def _assert_state_unchanged(profile, before):
    for name, value in before.items():
        current = getattr(profile, name).detach()
        if not torch.equal(current, value):
            raise RuntimeError(f"read-only diagnostic mutated profile state: {name}")


def run_diagnostic(args: argparse.Namespace) -> dict:
    project_root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    checkpoint_path = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(args.seed), workers=True)
    cfg = build_config(
        project_root,
        int(args.seed),
        data_root,
        int(args.batch_size),
        int(args.max_size),
        bool(args.fixed_eval_transform),
    )
    model = build_model(cfg)
    load_checkpoint(model, checkpoint_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    model.to(device)
    model.eval()
    model.set_current_epoch(int(args.epoch))
    model.set_total_epochs(int(args.max_epochs))

    profile = getattr(model.loss, "profile", None)
    if profile is None:
        raise ValueError("gradient conflict diagnostic requires a UniGIR checkpoint")
    profile.eval()
    before = _state_snapshot(profile)

    data_module = instantiate(
        cfg.data.data_module,
        model="InfMasking",
        data_root=str(data_root),
        batch_size=int(args.batch_size),
        num_workers=0,
        max_size=int(args.max_size),
        biased=True,
        seed=int(args.seed),
    )
    loader = data_module.train_dataloader()
    parameters, groups = _parameter_groups(model.named_parameters())
    records = []

    for batch_index, batch in enumerate(loader):
        if batch_index >= int(args.num_batches):
            break
        inputs = move_inputs(batch, device)
        outputs = model(*inputs)
        components = model.loss(outputs, return_components=True)
        base_loss = components["loss_base_raw"]
        profile_loss = components["loss_profile_raw"]
        if not profile_loss.requires_grad:
            raise RuntimeError("profile loss is not attached to the computation graph")

        base_gradients = torch.autograd.grad(
            base_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        profile_gradients = torch.autograd.grad(
            profile_loss,
            parameters,
            retain_graph=False,
            allow_unused=True,
        )
        row = {"batch": batch_index}
        for group, indices in groups.items():
            base_vector = _flatten_grads(parameters, base_gradients, indices)
            profile_vector = _flatten_grads(parameters, profile_gradients, indices)
            row[group] = gradient_metrics(base_vector, profile_vector)
        records.append(row)
        _assert_state_unchanged(profile, before)
        del components, outputs, base_gradients, profile_gradients

    if not records:
        raise RuntimeError("no batches were available for the gradient diagnostic")
    _assert_state_unchanged(profile, before)
    payload = {
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "max_size": int(args.max_size),
        "num_batches": len(records),
        "epoch": int(args.epoch),
        "max_epochs": int(args.max_epochs),
        "profile_loss_weight": float(profile.loss_weight),
        "parameter_groups": {key: len(value) for key, value in groups.items()},
        "records": records,
        "aggregate": summarize_records(records),
        "state_unchanged": True,
    }
    output_path = output_dir / "gradient_conflict.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"GRADIENT_DIAGNOSTIC_BATCHES={len(records)}")
    print(f"ALL_CONFLICT_FRACTION={payload['aggregate']['all']['conflict_fraction']:.6f}")
    print(f"JSON_OUT={output_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-eval-transform", action="store_true")
    parser.add_argument("--project-root", default="")
    args = parser.parse_args()
    run_diagnostic(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
