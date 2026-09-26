"""Diagnose a held-out non-additive fusion residual.

The residual is an operational proxy.  A ridge model is fitted on training
pairs to predict the complete fusion representation from both unimodal
representations.  The held-out residual is the part not explained by that
linear baseline; it is not claimed to be the information-theoretic synergy
component.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
from sklearn.linear_model import Ridge

from run_scripts.diagnose_fusion_relation_structure import (
    TOP_KS,
    _to_device,
    build_config,
    build_model,
    compute_metrics,
    load_checkpoint,
)


DEFAULT_RATIOS = (0.0, 0.7, 0.9)
DEFAULT_MASK_SEEDS = (101, 202, 303)


def concatenate_modalities(mod1: np.ndarray, mod2: np.ndarray) -> np.ndarray:
    """Build the linear baseline input from two aligned unimodal arrays."""
    first = np.asarray(mod1, dtype=np.float32)
    second = np.asarray(mod2, dtype=np.float32)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("unimodal arrays must be rank-2 with equal sample counts")
    return np.concatenate([first, second], axis=1)


def residual_from_prediction(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return target minus baseline prediction with shape validation."""
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must have the same rank-2 shape")
    return target - prediction


def set_mask_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_full(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    batches = {"mod1": [], "mod2": [], "fusion": []}
    with torch.inference_mode():
        for batch in loader:
            pair_images = _to_device(batch[0], device)
            z, _ = model.encoder(pair_images, mask_modalities=model.gen_all_possible_masks(2))
            values = {"mod1": z[0], "mod2": z[1], "fusion": z[-1]}
            for key, value in values.items():
                batches[key].append(value.detach().cpu())
    return {key: torch.cat(values, dim=0).numpy() for key, values in batches.items()}


def _collect_test_full_and_masked(model, loader, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    full_batches = {"mod1": [], "mod2": [], "fusion": []}
    masked_batches: dict[str, list[list[torch.Tensor]]] = {
        "mod1": [], "mod2": [], "fusion": []
    }
    with torch.inference_mode():
        for batch in loader:
            pair_images = _to_device(batch[0], device)
            z, mask_outputs = model.encoder(
                pair_images,
                mask_modalities=model.gen_all_possible_masks(2),
            )
            full_values = {"mod1": z[0], "mod2": z[1], "fusion": z[-1]}
            for key, value in full_values.items():
                full_batches[key].append(value.detach().cpu())
            branch_masks = {"mod1": mask_outputs[0], "mod2": mask_outputs[1], "fusion": mask_outputs[-1]}
            if not masked_batches["fusion"]:
                masked_batches = {key: [[] for _ in branch_masks[key]] for key in branch_masks}
            for key, views in branch_masks.items():
                for view_index, value in enumerate(views):
                    masked_batches[key][view_index].append(value.detach().cpu())
    full = {key: torch.cat(values, dim=0).numpy() for key, values in full_batches.items()}
    masked = {
        key: [torch.cat(chunks, dim=0).numpy() for chunks in view_batches]
        for key, view_batches in masked_batches.items()
    }
    return full, masked


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    project_root = Path(args.project_root).resolve() if args.project_root else PROJECT_ROOT
    checkpoint_path = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = tuple(float(value) for value in args.mask_ratios)
    mask_seeds = tuple(int(value) for value in args.mask_seeds)

    seed_everything(int(args.model_seed), workers=True)
    cfg = build_config(project_root, "baseline", int(args.model_seed), data_root, int(args.batch_size))
    model = build_model(cfg)
    load_checkpoint(model, checkpoint_path)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    cfg.data.data_module.max_size = int(args.train_samples)
    train_module = instantiate(cfg.data.data_module, model="Sup", biased=False, task="synergy")
    train_full = _collect_full(model, train_module.train_dataloader(), device)

    cfg.data.data_module.max_size = int(args.test_samples)
    test_module = instantiate(cfg.data.data_module, model="Sup", biased=False, task="synergy")
    test_loader = test_module.test_dataloader()
    sample_count = min(int(args.test_samples), len(test_module.val_dataset))

    features_train = concatenate_modalities(train_full["mod1"], train_full["mod2"])
    ridge = Ridge(alpha=float(args.ridge_alpha), fit_intercept=True)
    ridge.fit(features_train, train_full["fusion"])
    train_prediction = ridge.predict(features_train)
    train_residual = residual_from_prediction(train_full["fusion"], train_prediction)
    train_r2 = float(ridge.score(features_train, train_full["fusion"]))

    test_full, _ = _collect_test_full_and_masked(model, test_loader, device)
    test_features = concatenate_modalities(test_full["mod1"], test_full["mod2"])
    test_prediction = ridge.predict(test_features)
    full_residual = residual_from_prediction(test_full["fusion"], test_prediction)
    transformer = model.encoder.fusion_transformer
    rows: list[dict] = []
    for ratio in ratios:
        transformer.mask_ratio = ratio
        for mask_seed in mask_seeds:
            set_mask_seed(mask_seed)
            _, masked_views = _collect_test_full_and_masked(model, test_loader, device)
            for view_index in range(len(masked_views["fusion"])):
                masked_features = concatenate_modalities(
                    masked_views["mod1"][view_index], masked_views["mod2"][view_index]
                )
                masked_residual = residual_from_prediction(
                    masked_views["fusion"][view_index], ridge.predict(masked_features)
                )
                baseline_metrics = compute_metrics(test_full["fusion"], masked_views["fusion"][view_index])
                residual_metrics = compute_metrics(full_residual, masked_residual)
                rows.append({
                    "mask_ratio": ratio,
                    "mask_seed": mask_seed,
                    "mask_view": view_index,
                    "num_samples": sample_count,
                    "train_r2": train_r2,
                    "train_residual_norm_mean": float(np.linalg.norm(train_residual, axis=1).mean()),
                    "full_residual_norm_mean": float(np.linalg.norm(full_residual, axis=1).mean()),
                    "masked_residual_norm_mean": float(np.linalg.norm(masked_residual, axis=1).mean()),
                    **{f"fusion_{key}": value for key, value in baseline_metrics.items()},
                    **{f"residual_{key}": value for key, value in residual_metrics.items()},
                })
            print(f"DONE ratio={ratio:.3f} mask_seed={mask_seed} views={len(masked_views['fusion'])}")

    metric_names = [
        "fusion_spearman_relation", "fusion_self_cosine", "fusion_top5_retention",
        "residual_spearman_relation", "residual_self_cosine", "residual_top5_retention",
        "masked_residual_norm_mean",
    ]
    summary_rows = []
    for ratio in ratios:
        matching = [row for row in rows if row["mask_ratio"] == ratio]
        summary = {"mask_ratio": ratio, "num_rows": len(matching), "train_r2": train_r2}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in matching], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    payload = {
        "diagnostic": "nonadditive_residual_structure",
        "interpretation_boundary": "The held-out ridge residual is a non-additive proxy, not a direct information-theoretic synergy component.",
        "model": "InfMasking baseline checkpoint",
        "model_seed": int(args.model_seed),
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "pair_protocol": "Sup + biased=false + task=synergy",
        "fixed_eval_transform": "Resize(256)+CenterCrop(224)+ToTensor+ImageNetNormalize",
        "train_samples": len(train_full["fusion"]),
        "test_samples": sample_count,
        "ridge_alpha": float(args.ridge_alpha),
        "train_r2": train_r2,
        "mask_ratios": list(ratios),
        "mask_seeds": list(mask_seeds),
        "definition": "fit h_ab from [h_a;h_b] on train pairs, then compare held-out residual relations",
        "summary": summary_rows,
        "rows": rows,
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_dir / "residual_metrics.csv", rows)
    _write_csv(output_dir / "residual_summary.csv", summary_rows)
    print(f"EXPORTED_SAMPLES={sample_count}")
    print(f"EXPORTED_ROWS={len(rows)}")
    print(f"TRAIN_R2={train_r2:.6f}")
    print(f"JSON_OUT={output_dir / 'diagnostics.json'}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument("--mask-seeds", type=int, nargs="+", default=list(DEFAULT_MASK_SEEDS))
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
