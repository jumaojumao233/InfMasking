"""Aggregate three seed representation/profile diagnostic summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


TEXTURES = [
    "solid", "stripes", "grid", "hexgrid", "dots",
    "noise", "triangles", "zigzags", "rain", "pluses",
]


def _mean(values: list[float]) -> float:
    return mean(values)


def _summary_values(groups: list[dict], key: str) -> dict:
    values = [group[key]["mean"] for group in groups if key in group]
    if not values:
        return {}
    return {
        "mean": _mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "seed_values": values,
    }


def _delta_values(groups: list[dict], key: str) -> dict:
    values = [group[key]["delta"]["mean"] for group in groups if key in group]
    if not values:
        return {}
    return {
        "mean": _mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "seed_values": values,
    }


def aggregate(unigir_summaries: list[dict], comparisons: list[dict]) -> dict:
    if len(unigir_summaries) != 3 or len(comparisons) != 3:
        raise ValueError("expected exactly three UniGIR summaries and three comparisons")
    by_texture2 = {}
    for texture in TEXTURES:
        profile_groups = [summary["by_texture2"][texture] for summary in unigir_summaries]
        comparison_groups = [summary["by_texture2"][texture] for summary in comparisons]
        by_texture2[texture] = {
            "counts": [group["count"] for group in profile_groups],
            "profile_pred_accuracy": _summary_values(profile_groups, "profile_pred_accuracy"),
            "profile_kl": _summary_values(profile_groups, "profile_kl"),
            "profile_target_confidence": _summary_values(profile_groups, "profile_target_confidence"),
            "profile_target_entropy": _summary_values(profile_groups, "profile_target_entropy"),
            "masked_to_full_distance_mean": _delta_values(comparison_groups, "masked_to_full_distance_mean"),
        }
    return {
        "seeds": [summary["seed"] for summary in unigir_summaries],
        "num_samples_per_seed": [summary["num_samples"] for summary in unigir_summaries],
        "by_texture2": by_texture2,
    }


def _value(group: dict, key: str) -> str:
    item = group.get(key, {})
    if not item:
        return "—"
    return f"{item['mean']:.4f}"


def render(summary: dict) -> str:
    seeds = "、".join(str(seed) for seed in summary["seeds"])
    lines = [
        "# G0 三 seed 逐样本 profile 与表示诊断汇总",
        "",
        f"UniGIR seed：{seeds}；每个 seed 的样本数：{summary['num_samples_per_seed']}。profile 指标为 UniGIR 三 seed 均值；masked distance 差值为 `UniGIR - Baseline`。",
        "固定 test transform 下两次完整视图相同，因此没有把 full view distance 当作分析指标。",
        "",
        "| texture2 | 样本数（42/7/123） | profile acc 均值 | profile KL 均值 | target confidence 均值 | target entropy 均值 | masked distance 差值均值 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for texture in TEXTURES:
        group = summary["by_texture2"][texture]
        counts = "/".join(str(value) for value in group["counts"])
        lines.append(
            f"| {texture} | {counts} | {_value(group, 'profile_pred_accuracy')} | {_value(group, 'profile_kl')} | {_value(group, 'profile_target_confidence')} | {_value(group, 'profile_target_entropy')} | {_value(group, 'masked_to_full_distance_mean')} |"
        )
    lines.extend([
        "",
        "表中 profile acc、profile KL 和表示距离都只是描述性统计。三 seed 的样本量相同，均值可以直接比较；没有进行显著性检验。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unigir-summary", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.unigir_summary]
    comparisons = [json.loads(path.read_text(encoding="utf-8")) for path in args.comparison]
    result = aggregate(summaries, comparisons)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_out.write_text(render(result), encoding="utf-8")
    print(f"AGGREGATED_SEEDS={len(result['seeds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
