"""Compare fusion, unimodal, and interaction-increment relation structure.

The interaction signature in this diagnostic is an operational proxy.  It is
the change from each unimodal encoder branch to the complete fusion branch,
not a claim that the resulting vector is the information-theoretic synergy
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
import torch.nn.functional as F
from hydra.utils import instantiate
from pytorch_lightning import seed_everything

from run_scripts.diagnose_fusion_relation_structure import (
    TOP_KS,
    _to_device,
    build_config,
    build_model,
    compute_metrics,
    load_checkpoint,
)


DEFAULT_RATIOS = (0.0, 0.5, 0.7, 0.9)
DEFAULT_MASK_SEEDS = (101, 202, 303)


def interaction_signature(full: torch.Tensor, mod1: torch.Tensor, mod2: torch.Tensor) -> torch.Tensor:
    """Represent the two modality addition increments in a common vector."""
    delta1 = F.normalize(full - mod1, p=2, dim=-1, eps=1e-12)
    delta2 = F.normalize(full - mod2, p=2, dim=-1, eps=1e-12)
    return torch.cat([delta1, delta2], dim=-1)


def set_mask_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manifest_rows(dataset, count: int) -> list[dict]:
    rows = []
    attr_to_id = {"shape": 0, "color": 1, "texture": 2}
    for sample_index in range(count):
        idx1, idx2 = dataset.idx_pairs[sample_index]
        target1 = dataset.target_transform(dataset.name_images[int(idx1)])
        target2 = dataset.target_transform(dataset.name_images[int(idx2)])
        synergy_pair = (
            int(target1[attr_to_id[dataset.synergy_attr[0]]]),
            int(target2[attr_to_id[dataset.synergy_attr[1]]]),
        )
        rows.append({
            "sample_index": sample_index,
            "base_index_1": int(idx1),
            "base_index_2": int(idx2),
            "shape_1": int(target1[attr_to_id["shape"]]),
            "shape_2": int(target2[attr_to_id["shape"]]),
            "texture_1": int(target1[attr_to_id["texture"]]),
            "texture_2": int(target2[attr_to_id["texture"]]),
            "color_1": int(target1[attr_to_id["color"]]),
            "color_2": int(target2[attr_to_id["color"]]),
            "synergy_pair": list(synergy_pair),
            "synergy_target": int(synergy_pair in dataset.correlated_feature_pairs),
        })
    return rows


def _collect_full(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    batches = {"fusion": [], "mod1": [], "mod2": [], "interaction": []}
    with torch.inference_mode():
        for batch in loader:
            pair_images = _to_device(batch[0], device)
            z, _ = model.encoder(pair_images, mask_modalities=model.gen_all_possible_masks(2))
            fusion, mod1, mod2 = z[-1], z[0], z[1]
            values = {
                "fusion": fusion,
                "mod1": mod1,
                "mod2": mod2,
                "interaction": interaction_signature(fusion, mod1, mod2),
            }
            for key, value in values.items():
                batches[key].append(value.detach().cpu())
    return {key: torch.cat(value, dim=0).numpy() for key, value in batches.items()}


def _collect_masked(model, loader, device: torch.device) -> dict[str, list[np.ndarray]]:
    batches: dict[str, list[list[torch.Tensor]]] = {
        "fusion": [], "mod1": [], "mod2": [], "interaction": []
    }
    with torch.inference_mode():
        for batch in loader:
            pair_images = _to_device(batch[0], device)
            z, mask_outputs = model.encoder(
                pair_images,
                mask_modalities=model.gen_all_possible_masks(2),
            )
            fusion_masks, mod1_masks, mod2_masks = mask_outputs[-1], mask_outputs[0], mask_outputs[1]
            if not batches["fusion"]:
                num_views = len(fusion_masks)
                batches = {key: [[] for _ in range(num_views)] for key in batches}
            values = {
                "fusion": fusion_masks,
                "mod1": mod1_masks,
                "mod2": mod2_masks,
                "interaction": [
                    interaction_signature(fusion_masks[index], mod1_masks[index], mod2_masks[index])
                    for index in range(len(fusion_masks))
                ],
            }
            for key, views in values.items():
                for view_index, value in enumerate(views):
                    batches[key][view_index].append(value.detach().cpu())
    return {
        key: [torch.cat(chunks, dim=0).numpy() for chunks in view_batches]
        for key, view_batches in batches.items()
    }


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
    cfg.data.data_module.max_size = int(args.max_samples)
    model = build_model(cfg)
    load_checkpoint(model, checkpoint_path)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    data_module = instantiate(
        cfg.data.data_module,
        model="Sup",
        biased=False,
        task="synergy",
    )
    loader = data_module.test_dataloader()
    dataset = data_module.val_dataset
    sample_count = min(int(args.max_samples), len(dataset))
    full = _collect_full(model, loader, device)
    manifest = _manifest_rows(dataset, sample_count)
    if any(len(values) != sample_count for values in full.values()):
        raise RuntimeError("full representation sample count mismatch")

    transformer = model.encoder.fusion_transformer
    rows: list[dict] = []
    for ratio in ratios:
        transformer.mask_ratio = ratio
        for mask_seed in mask_seeds:
            set_mask_seed(mask_seed)
            masked = _collect_masked(model, loader, device)
            for view_index in range(len(masked["fusion"])):
                for kind in ("fusion", "mod1", "mod2", "interaction"):
                    metrics = compute_metrics(full[kind], masked[kind][view_index])
                    rows.append({
                        "mask_ratio": ratio,
                        "mask_seed": mask_seed,
                        "mask_view": view_index,
                        "representation": kind,
                        "num_samples": sample_count,
                        **metrics,
                    })
            print(f"DONE ratio={ratio:.3f} mask_seed={mask_seed} views={len(masked['fusion'])}")

    metric_names = ["spearman_relation", "self_cosine"] + [f"top{k}_retention" for k in TOP_KS]
    summary_rows = []
    for ratio in ratios:
        for kind in ("fusion", "mod1", "mod2", "interaction"):
            matching = [
                row for row in rows
                if row["mask_ratio"] == ratio and row["representation"] == kind
            ]
            summary = {"mask_ratio": ratio, "representation": kind, "num_rows": len(matching)}
            for metric in metric_names:
                values = np.asarray([row[metric] for row in matching], dtype=np.float64)
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary_rows.append(summary)

    payload = {
        "diagnostic": "interaction_signature_structure",
        "interpretation_boundary": "The interaction signature is an operational increment proxy, not a direct information-theoretic synergy component.",
        "model": "InfMasking baseline checkpoint",
        "model_seed": int(args.model_seed),
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "pair_protocol": "Sup + biased=false + task=synergy",
        "fixed_eval_transform": "Resize(256)+CenterCrop(224)+ToTensor+ImageNetNormalize",
        "num_samples": sample_count,
        "mask_ratios": list(ratios),
        "mask_seeds": list(mask_seeds),
        "representations": {
            "fusion": "unprojected complete two-modality fusion output",
            "mod1": "unprojected modality 1 branch output",
            "mod2": "unprojected modality 2 branch output",
            "interaction": "concatenated normalized increments h_ab-h_a and h_ab-h_b",
        },
        "sample_manifest": manifest,
        "summary": summary_rows,
        "rows": rows,
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_dir / "interaction_metrics.csv", rows)
    _write_csv(output_dir / "interaction_summary.csv", summary_rows)
    print(f"EXPORTED_SAMPLES={sample_count}")
    print(f"EXPORTED_ROWS={len(rows)}")
    print(f"JSON_OUT={output_dir / 'diagnostics.json'}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument("--mask-seeds", type=int, nargs="+", default=list(DEFAULT_MASK_SEEDS))
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
