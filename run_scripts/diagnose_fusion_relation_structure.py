"""Diagnose fusion relational structure under test-time token masking.

This is a read-only diagnostic.  It loads one fixed checkpoint, keeps one
deterministic G0 pair set for every masking ratio, and measures how the
pairwise relation matrix of the complete fusion projection changes for each
random masked view.  The relation matrix is an operational proxy for fusion
relational structure; it is not a direct measurement of synergy structure.
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
from scipy.stats import spearmanr

from pytorch_lightning import seed_everything

from run_scripts.export_profile_representation_diagnostics import (
    _pair_labels,
    _to_device,
    build_config,
    build_model,
    load_checkpoint,
)


DEFAULT_RATIOS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_MASK_SEEDS = (101, 202, 303, 404, 505)
TOP_KS = (5, 10, 20)


def relation_upper_triangle(relation: np.ndarray) -> np.ndarray:
    """Return non-diagonal upper-triangle relation values in row-major order."""
    matrix = np.asarray(relation, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"relation must be square, got shape={matrix.shape}")
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def relation_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute a cosine relation matrix from one embedding per sample."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"embeddings must be rank 2, got shape={values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero-norm embedding encountered")
    normalized = values / norms
    return normalized @ normalized.T


def spearman_relation(full_relation: np.ndarray, masked_relation: np.ndarray) -> float:
    """Measure rank preservation of two square relation matrices."""
    full_values = relation_upper_triangle(full_relation)
    masked_values = relation_upper_triangle(masked_relation)
    if full_values.size < 2:
        raise ValueError("at least two pairwise relations are required")
    full_constant = np.allclose(full_values, full_values[0])
    masked_constant = np.allclose(masked_values, masked_values[0])
    if full_constant or masked_constant:
        return 1.0 if np.allclose(full_values, masked_values) else 0.0
    result = spearmanr(full_values, masked_values)
    value = float(result.statistic)
    return value if np.isfinite(value) else 0.0


def topk_neighbor_retention(
    full_relation: np.ndarray,
    masked_relation: np.ndarray,
    k: int,
) -> float:
    """Average the per-sample overlap of the full and masked top-k neighbors."""
    full = np.asarray(full_relation, dtype=np.float64).copy()
    masked = np.asarray(masked_relation, dtype=np.float64).copy()
    if full.shape != masked.shape or full.ndim != 2 or full.shape[0] != full.shape[1]:
        raise ValueError("full and masked relations must have the same square shape")
    n_samples = full.shape[0]
    if not 1 <= k < n_samples:
        raise ValueError(f"k must be in [1, {n_samples - 1}], got {k}")
    np.fill_diagonal(full, -np.inf)
    np.fill_diagonal(masked, -np.inf)
    full_neighbors = np.argpartition(full, -k, axis=1)[:, -k:]
    masked_neighbors = np.argpartition(masked, -k, axis=1)[:, -k:]
    overlaps = [
        len(set(full_row.tolist()).intersection(masked_row.tolist())) / float(k)
        for full_row, masked_row in zip(full_neighbors, masked_neighbors)
    ]
    return float(np.mean(overlaps))


def sample_self_cosine(full_embeddings: np.ndarray, masked_embeddings: np.ndarray) -> float:
    """Average cosine between each sample's full and masked representations."""
    full = np.asarray(full_embeddings, dtype=np.float64)
    masked = np.asarray(masked_embeddings, dtype=np.float64)
    if full.shape != masked.shape or full.ndim != 2:
        raise ValueError("full and masked embeddings must have the same rank-2 shape")
    full_norm = full / np.linalg.norm(full, axis=1, keepdims=True)
    masked_norm = masked / np.linalg.norm(masked, axis=1, keepdims=True)
    return float(np.mean(np.sum(full_norm * masked_norm, axis=1)))


def compute_metrics(full_embeddings: np.ndarray, masked_embeddings: np.ndarray) -> dict[str, float]:
    full_relation = relation_matrix(full_embeddings)
    masked_relation = relation_matrix(masked_embeddings)
    metrics = {
        "spearman_relation": spearman_relation(full_relation, masked_relation),
        "self_cosine": sample_self_cosine(full_embeddings, masked_embeddings),
    }
    for k in TOP_KS:
        metrics[f"top{k}_retention"] = topk_neighbor_retention(full_relation, masked_relation, k)
    return metrics


def set_mask_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_full_embeddings(model, loader, device: torch.device) -> np.ndarray:
    batches = []
    with torch.inference_mode():
        for batch in loader:
            pair_images = _to_device(batch[0], device)
            outputs = model(pair_images, [value.clone() for value in pair_images])
            full = outputs["aug1_embed"][outputs["prototype"]]
            batches.append(full.detach().cpu())
    if not batches:
        raise ValueError("the fixed diagnostic loader returned no samples")
    return torch.cat(batches, dim=0).numpy()


def _collect_masked_embeddings(model, loader, device: torch.device) -> list[np.ndarray]:
    views: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for batch in loader:
            pair_images = _to_device(batch[0], device)
            outputs = model(pair_images, [value.clone() for value in pair_images])
            masked = outputs["mask_out1"][outputs["prototype"]]
            if not views:
                views = [[] for _ in masked]
            if len(views) != len(masked):
                raise RuntimeError("number of masked views changed within one run")
            for view_index, value in enumerate(masked):
                views[view_index].append(value.detach().cpu())
    if not views:
        raise ValueError("the model returned no masked views")
    return [torch.cat(chunks, dim=0).numpy() for chunks in views]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
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
    if any(value < 0.0 or value > 1.0 for value in ratios):
        raise ValueError(f"mask ratios must be in [0, 1], got {ratios}")
    if not mask_seeds:
        raise ValueError("at least one mask seed is required")

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
        task="unique2",
    )
    loader = data_module.test_dataloader()
    dataset = data_module.val_dataset
    expected_samples = min(int(args.max_samples), len(dataset))

    transformer = model.encoder.fusion_transformer
    if not transformer.random_mask:
        raise ValueError("the checkpoint model has random_mask=False")

    # Full embeddings and the pair manifest are fixed once and reused for all
    # ratios and mask seeds.  This prevents data resampling from entering the
    # relation-preservation comparison.
    full_embeddings = _collect_full_embeddings(model, loader, device)
    if len(full_embeddings) != expected_samples:
        raise RuntimeError(f"sample count changed: {len(full_embeddings)} != {expected_samples}")
    full_relation = relation_matrix(full_embeddings)
    sample_manifest = _pair_labels(dataset, 0, expected_samples)

    rows: list[dict] = []
    for ratio in ratios:
        transformer.mask_ratio = ratio
        for mask_seed in mask_seeds:
            set_mask_seed(mask_seed)
            masked_views = _collect_masked_embeddings(model, loader, device)
            for view_index, masked_embeddings in enumerate(masked_views):
                if len(masked_embeddings) != len(full_embeddings):
                    raise RuntimeError("full and masked sample counts do not match")
                metrics = compute_metrics(full_embeddings, masked_embeddings)
                rows.append({
                    "mask_ratio": ratio,
                    "mask_seed": mask_seed,
                    "mask_view": view_index,
                    "num_samples": len(full_embeddings),
                    **metrics,
                })
            print(f"DONE ratio={ratio:.3f} mask_seed={mask_seed} views={len(masked_views)}")

    summary_rows = []
    metric_names = ["spearman_relation", "self_cosine"] + [f"top{k}_retention" for k in TOP_KS]
    for ratio in ratios:
        matching = [row for row in rows if row["mask_ratio"] == ratio]
        summary = {"mask_ratio": ratio, "num_rows": len(matching)}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in matching], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    payload = {
        "diagnostic": "fusion_relation_structure",
        "interpretation_boundary": "Fusion relation structure is an operational proxy; it is not a direct measurement of synergy structure.",
        "model": "InfMasking baseline checkpoint",
        "model_seed": int(args.model_seed),
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "pair_protocol": "Sup + biased=false + task=unique2",
        "fixed_eval_transform": "Resize(256)+CenterCrop(224)+ToTensor+ImageNetNormalize",
        "num_samples": len(full_embeddings),
        "mask_ratios": list(ratios),
        "mask_seeds": list(mask_seeds),
        "num_masked_views": len(rows) // (len(ratios) * len(mask_seeds)),
        "full_relation_definition": "cosine similarity of normalized complete fusion projections",
        "masked_relation_definition": "cosine similarity of normalized masked fusion projections",
        "metrics": ["Spearman relation correlation", "Top-k neighbor retention k=5,10,20", "full-mask self cosine"],
        "sample_manifest": sample_manifest,
        "summary": summary_rows,
        "rows": rows,
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_dir / "relation_metrics.csv", rows)
    _write_csv(output_dir / "relation_summary.csv", summary_rows)
    np.save(output_dir / "full_relation.npy", full_relation)
    np.save(output_dir / "full_embeddings.npy", full_embeddings)
    print(f"EXPORTED_SAMPLES={len(full_embeddings)}")
    print(f"EXPORTED_ROWS={len(rows)}")
    print(f"JSON_OUT={output_dir / 'diagnostics.json'}")
    print(f"SUMMARY_OUT={output_dir / 'relation_summary.csv'}")
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
