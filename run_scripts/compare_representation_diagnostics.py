"""Compare paired per-sample representation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


TEXTURES = [
    "solid", "stripes", "grid", "hexgrid", "dots",
    "noise", "triangles", "zigzags", "rain", "pluses",
]

REPRESENTATION_KEYS = (
    "full_aug_distance",
    "masked_to_full_distance_a",
    "masked_to_full_distance_b",
    "masked_to_full_distance_mean",
)


def _stats(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def compare(baseline: dict, unigir: dict) -> dict:
    base_rows = baseline.get("rows", [])
    uni_rows = unigir.get("rows", [])
    if len(base_rows) != len(uni_rows):
        raise ValueError("paired exports have different sample counts")
    for base, uni in zip(base_rows, uni_rows):
        keys = ("sample_index", "base_index_1", "base_index_2", "texture1", "texture2")
        if any(base[key] != uni[key] for key in keys):
            raise ValueError("paired exports do not have identical sample order")

    overall = {}
    by_texture2 = {}
    for texture in TEXTURES:
        by_texture2[texture] = {"count": 0}
    for key in REPRESENTATION_KEYS:
        deltas = [float(uni[key]) - float(base[key]) for base, uni in zip(base_rows, uni_rows)]
        overall[key] = {
            "baseline": _stats([float(row[key]) for row in base_rows]),
            "unigir": _stats([float(row[key]) for row in uni_rows]),
            "delta": _stats(deltas),
        }
    for texture in TEXTURES:
        selected = [(base, uni) for base, uni in zip(base_rows, uni_rows) if uni["texture2"] == texture]
        by_texture2[texture]["count"] = len(selected)
        for key in REPRESENTATION_KEYS:
            base_values = [float(base[key]) for base, _ in selected]
            uni_values = [float(uni[key]) for _, uni in selected]
            deltas = [new - old for old, new in zip(base_values, uni_values)]
            if deltas:
                by_texture2[texture][key] = {
                    "baseline": _stats(base_values),
                    "unigir": _stats(uni_values),
                    "delta": _stats(deltas),
                }
    return {
        "seed": baseline["seed"],
        "num_samples": len(base_rows),
        "baseline_profile_available": baseline["profile_available"],
        "unigir_profile_available": unigir["profile_available"],
        "overall": overall,
        "by_texture2": by_texture2,
    }


def _value(group: dict, key: str, field: str) -> str:
    item = group.get(key)
    if item is None:
        return "—"
    return f"{item[field]['mean']:.4f}"


def render(summary: dict) -> str:
    lines = [
        f"# G0 seed={summary['seed']} Baseline/UniGIR 表示诊断对比",
        "",
        f"样本数：{summary['num_samples']}。所有差值按 `UniGIR - Baseline` 计算；表示距离定义为 `1 - cosine`。",
        "固定 test transform 会让两次视图完全相同，因此 `full_aug_distance` 只作为一致性检查，主要看 masked-to-full 距离。",
        "",
        "## 按 unique2 目标纹理的表示距离",
        "",
        "| texture2 | 样本数 | Baseline masked distance | UniGIR masked distance | 差值 | Baseline full distance | UniGIR full distance | 差值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for texture in TEXTURES:
        group = summary["by_texture2"][texture]
        lines.append(
            f"| {texture} | {group['count']} | {_value(group, 'masked_to_full_distance_mean', 'baseline')} | {_value(group, 'masked_to_full_distance_mean', 'unigir')} | {_value(group, 'masked_to_full_distance_mean', 'delta')} | {_value(group, 'full_aug_distance', 'baseline')} | {_value(group, 'full_aug_distance', 'unigir')} | {_value(group, 'full_aug_distance', 'delta')} |"
        )
    lines.extend(["", "## 总体表示距离", "", "| 指标 | Baseline | UniGIR | UniGIR - Baseline |", "|---|---:|---:|---:|"])
    for key in REPRESENTATION_KEYS:
        group = summary["overall"][key]
        lines.append(f"| `{key}` | {group['baseline']['mean']:.4f} | {group['unigir']['mean']:.4f} | {group['delta']['mean']:+.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--unigir-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    unigir = json.loads(args.unigir_json.read_text(encoding="utf-8"))
    summary = compare(baseline, unigir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.markdown_out.write_text(render(summary), encoding="utf-8")
    print(f"COMPARED_SAMPLES={summary['num_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
