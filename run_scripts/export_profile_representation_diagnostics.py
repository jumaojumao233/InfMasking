"""Export per-sample profile and representation diagnostics from a checkpoint.

This script is deliberately separate from the training and Lightning test
entrypoints.  It loads one checkpoint, runs the fixed ``Sup + biased=false``
unique2 test loader in order, and writes one row per pair.  It never calls
``trainer.fit`` and never updates the profile prototypes or feature queue
because the model stays in eval mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything


TEXTURES = [
    "solid",
    "stripes",
    "grid",
    "hexgrid",
    "dots",
    "noise",
    "triangles",
    "zigzags",
    "rain",
    "pluses",
]

REPRESENTATION_KEYS = (
    "full_aug_cosine",
    "full_aug_distance",
    "masked_to_full_distance_a",
    "masked_to_full_distance_b",
    "masked_to_full_distance_mean",
)

PROFILE_KEYS = (
    "profile_pred_accuracy",
    "profile_kl",
    "profile_target_confidence",
    "profile_target_entropy",
    "profile_prediction_confidence",
)


def _project_root() -> Path:
    return PROJECT_ROOT


def build_config(project_root: Path, method: str, seed: int, data_root: Path,
                 batch_size: int) -> object:
    """Compose the same effective model/data config used by main_trifeatures."""
    base = OmegaConf.load(project_root / "configs" / "train_trifeatures.yaml")
    model_name = "infmasking" if method == "baseline" else "unigir"
    model_group = OmegaConf.load(project_root / "configs" / "model" / f"{model_name}.yaml")
    data_group = OmegaConf.load(project_root / "configs" / "data" / "trifeatures.yaml")
    merged_model = OmegaConf.merge(base.model, model_group)
    cfg = OmegaConf.merge(base, {"model": merged_model, "data": data_group})

    cfg.seed = int(seed)
    cfg.mode = "test"
    cfg.enable_linear_probe = False
    cfg.data.data_module.data_root = str(data_root)
    cfg.data.data_module.fixed_eval_transform = True
    cfg.data.data_module.batch_size = int(batch_size)
    cfg.data.data_module.num_workers = 0

    # Match the G0 checkpoint architecture and the fixed evaluation command.
    cfg.model.model.encoder.embed_dim = 256
    cfg.model.adapters[0].dim_tokens = 256
    cfg.model.adapters[1].dim_tokens = 256
    cfg.model.model.encoder.num_mask = 3
    cfg.model.model.encoder.mask_ratio = 0.7
    if method == "unigir":
        cfg.model.model.loss_kwargs.profile_kwargs.num_prototypes = 128
        cfg.model.model.loss_kwargs.profile_kwargs.queue_size = 1024
        cfg.model.model.loss_kwargs.profile_kwargs.loss_weight = 0.25
        cfg.model.model.loss_kwargs.profile_kwargs.cross = False
        cfg.model.model.loss_kwargs.profile_kwargs.shuffle_targets = False
    OmegaConf.resolve(cfg)
    return cfg


def build_model(cfg: object) -> torch.nn.Module:
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


def _to_device(batch_part: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [value.to(device, non_blocking=True) for value in batch_part]


def _profile_rows(profile, z_full, masks) -> dict[str, np.ndarray]:
    """Return one vector per sample without changing profile buffers."""
    prot = profile.prototypes.detach()
    z_full = profile._norm(z_full)
    masks = [profile._norm(value) for value in masks]
    q = profile._codes(z_full, prot)
    logits = torch.stack(masks, dim=0) @ prot.t()
    log_p = F.log_softmax(logits / profile.temperature, dim=-1)
    p = log_p.exp()
    target_ids = q.argmax(dim=-1)
    pred_ids = logits.argmax(dim=-1)
    pred_accuracy = (pred_ids == target_ids.unsqueeze(0)).float().mean(dim=0)
    kl = (q.unsqueeze(0) * (q.clamp_min(1e-12).log().unsqueeze(0) - log_p)).sum(dim=-1).mean(dim=0)
    target_entropy = -(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(dim=-1)
    target_entropy = target_entropy / torch.log(torch.tensor(float(profile.num_prototypes), device=q.device))
    target_confidence = q.max(dim=-1).values
    prediction_confidence = p.mean(dim=0).max(dim=-1).values
    return {
        "profile_pred_accuracy": pred_accuracy.detach().cpu().numpy(),
        "profile_kl": kl.detach().cpu().numpy(),
        "profile_target_confidence": target_confidence.detach().cpu().numpy(),
        "profile_target_entropy": target_entropy.detach().cpu().numpy(),
        "profile_prediction_confidence": prediction_confidence.detach().cpu().numpy(),
        "profile_target_id": target_ids.detach().cpu().numpy(),
        "profile_prediction_id": p.mean(dim=0).argmax(dim=-1).detach().cpu().numpy(),
    }


def _representation_rows(z_a, z_b, masks_a, masks_b) -> dict[str, np.ndarray]:
    z_a = F.normalize(z_a, p=2, dim=-1)
    z_b = F.normalize(z_b, p=2, dim=-1)
    full_cosine = (z_a * z_b).sum(dim=-1)
    distances_a = [1.0 - (F.normalize(value, p=2, dim=-1) * z_a).sum(dim=-1) for value in masks_a]
    distances_b = [1.0 - (F.normalize(value, p=2, dim=-1) * z_b).sum(dim=-1) for value in masks_b]
    distance_a = torch.stack(distances_a, dim=0).mean(dim=0)
    distance_b = torch.stack(distances_b, dim=0).mean(dim=0)
    return {
        "full_aug_cosine": full_cosine.detach().cpu().numpy(),
        "full_aug_distance": (1.0 - full_cosine).detach().cpu().numpy(),
        "masked_to_full_distance_a": distance_a.detach().cpu().numpy(),
        "masked_to_full_distance_b": distance_b.detach().cpu().numpy(),
        "masked_to_full_distance_mean": ((distance_a + distance_b) / 2.0).detach().cpu().numpy(),
    }


def _pair_labels(dataset, start: int, count: int) -> list[dict]:
    rows = []
    attr_to_id = {"shape": 0, "color": 1, "texture": 2}
    for sample_index in range(start, start + count):
        idx1, idx2 = dataset.idx_pairs[sample_index]
        target1 = dataset.target_transform(dataset.name_images[int(idx1)])
        target2 = dataset.target_transform(dataset.name_images[int(idx2)])
        texture1 = int(target1[attr_to_id["texture"]])
        texture2 = int(target2[attr_to_id["texture"]])
        rows.append({
            "sample_index": sample_index,
            "base_index_1": int(idx1),
            "base_index_2": int(idx2),
            "texture1_id": texture1,
            "texture1": TEXTURES[texture1],
            "texture2_id": texture2,
            "texture2": TEXTURES[texture2],
            "unique2_target": texture2,
        })
    return rows


def export(args: argparse.Namespace) -> dict:
    project_root = Path(args.project_root).resolve() if args.project_root else _project_root()
    checkpoint_path = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(args.seed), workers=True)
    cfg = build_config(project_root, args.method, int(args.seed), data_root, int(args.batch_size))
    model = build_model(cfg)
    load_checkpoint(model, checkpoint_path)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Match the unique2 linear-probe protocol exactly.  The SSL training data
    # module uses biased=True and would produce a different pair set.
    data_module = instantiate(
        cfg.data.data_module,
        model="Sup",
        biased=False,
        task="unique2",
    )
    loader = data_module.test_dataloader()
    dataset = data_module.val_dataset
    profile = getattr(model.loss, "profile", None)
    rows: list[dict] = []
    sample_offset = 0

    with torch.inference_mode():
        for batch in loader:
            pair_images = _to_device(batch[0], device)
            # Sup returns one deterministic image pair and a unique2 label.
            # Feed the same pair twice so only the model's masked views differ.
            x1 = pair_images
            x2 = [value.clone() for value in pair_images]
            outputs = model(x1, x2)
            prototype = outputs["prototype"]
            z_a = outputs["aug1_embed"][prototype]
            z_b = outputs["aug2_embed"][prototype]
            masks_a = outputs["mask_out1"][prototype]
            masks_b = outputs["mask_out2"][prototype]
            batch_size = z_a.shape[0]
            batch_rows = _pair_labels(dataset, sample_offset, batch_size)
            batch_rep = _representation_rows(z_a, z_b, masks_a, masks_b)
            batch_profile = _profile_rows(profile, z_a, masks_a) if profile is not None else None
            if batch_profile is not None:
                profile_b = _profile_rows(profile, z_b, masks_b)
            else:
                profile_b = None
            for index, row in enumerate(batch_rows):
                for key, values in batch_rep.items():
                    row[key] = float(values[index])
                if batch_profile is None:
                    for key in PROFILE_KEYS:
                        row[key] = None
                    row["profile_target_id"] = None
                    row["profile_prediction_id"] = None
                    row["profile_target_id_b"] = None
                    row["profile_prediction_id_b"] = None
                else:
                    for key in PROFILE_KEYS:
                        row[key] = float((batch_profile[key][index] + profile_b[key][index]) / 2.0)
                    row["profile_target_id"] = int(batch_profile["profile_target_id"][index])
                    row["profile_prediction_id"] = int(batch_profile["profile_prediction_id"][index])
                    row["profile_target_id_b"] = int(profile_b["profile_target_id"][index])
                    row["profile_prediction_id_b"] = int(profile_b["profile_prediction_id"][index])
                rows.append(row)
            sample_offset += batch_size
            if args.max_samples and len(rows) >= args.max_samples:
                rows = rows[: args.max_samples]
                break

    payload = {
        "method": args.method,
        "seed": int(args.seed),
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "num_samples": len(rows),
        "fixed_eval_transform": "Resize(256)+CenterCrop(224)+ToTensor+ImageNetNormalize",
        "pair_protocol": "Sup + biased=false + task=unique2",
        "profile_available": profile is not None,
        "profile_state_is_eval_only": True,
        "rows": rows,
    }
    json_path = output_dir / "diagnostics.json"
    csv_path = output_dir / "diagnostics.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"EXPORTED_SAMPLES={len(rows)}")
    print(f"PROFILE_AVAILABLE={profile is not None}")
    print(f"JSON_OUT={json_path}")
    print(f"CSV_OUT={csv_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=("baseline", "unigir"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--project-root", default="")
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
