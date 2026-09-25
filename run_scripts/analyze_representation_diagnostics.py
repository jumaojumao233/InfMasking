"""Summarize per-sample profile and representation diagnostic exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


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

NUMERIC_KEYS = (
    "full_aug_cosine",
    "full_aug_distance",
    "masked_to_full_distance_a",
    "masked_to_full_distance_b",
    "masked_to_full_distance_mean",
    "profile_pred_accuracy",
    "profile_kl",
    "profile_target_confidence",
    "profile_target_entropy",
    "profile_prediction_confidence",
)


def _stats(rows: list[dict]) -> dict:
    result = {"count": len(rows)}
    for key in NUMERIC_KEYS:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            continue
        result[key] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return result


def summarize(payload: dict) -> dict:
    rows = payload.get("rows", [])
    by_texture2 = {
        texture: _stats([row for row in rows if row["texture2"] == texture])
        for texture in TEXTURES
    }
    by_texture1 = {
        texture: _stats([row for row in rows if row["texture1"] == texture])
        for texture in TEXTURES
    }
    return {
        "method": payload["method"],
        "seed": payload["seed"],
        "num_samples": payload["num_samples"],
        "profile_available": payload["profile_available"],
        "fixed_eval_transform": payload["fixed_eval_transform"],
        "pair_protocol": payload.get("pair_protocol", "未记录"),
        "overall": _stats(rows),
        "by_texture2": by_texture2,
        "by_texture1": by_texture1,
    }


def _value(group: dict, key: str) -> str:
    item = group.get(key)
    return "—" if item is None else f"{item['mean']:.4f}"


def render(summary: dict) -> str:
    lines = [
        f"# {summary['method']} seed={summary['seed']} 逐样本诊断",
        "",
        f"样本数：{summary['num_samples']}；profile 分支：{'有' if summary['profile_available'] else '无'}；评测变换：`{summary['fixed_eval_transform']}`。",
        f"pair 协议：`{summary['pair_protocol']}`。",
        "表中 `texture2` 是 unique2 的目标纹理；表示距离定义为 `1 - cosine`。profile 指标对两次增强取平均。",
        "",
        "## 总体统计",
        "",
        "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |",
        "|---|---:|---:|---:|---:|",
    ]
    overall = summary["overall"]
    for key in NUMERIC_KEYS:
        if key in overall:
            item = overall[key]
            lines.append(f"| `{key}` | {item['mean']:.4f} | {item['std']:.4f} | {item['min']:.4f} | {item['max']:.4f} |")

    lines.extend([
        "",
        "## 按 unique2 目标纹理汇总",
        "",
        "| texture2 | 样本数 | full distance | masked distance | profile acc | profile KL | target confidence | target entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for texture in TEXTURES:
        group = summary["by_texture2"][texture]
        lines.append(
            f"| {texture} | {group['count']} | {_value(group, 'full_aug_distance')} | {_value(group, 'masked_to_full_distance_mean')} | {_value(group, 'profile_pred_accuracy')} | {_value(group, 'profile_kl')} | {_value(group, 'profile_target_confidence')} | {_value(group, 'profile_target_entropy')} |"
        )
    lines.extend([
        "",
        "## 按第一张图纹理汇总",
        "",
        "| texture1 | 样本数 | full distance | masked distance | profile acc | profile KL | target confidence | target entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for texture in TEXTURES:
        group = summary["by_texture1"][texture]
        lines.append(
            f"| {texture} | {group['count']} | {_value(group, 'full_aug_distance')} | {_value(group, 'masked_to_full_distance_mean')} | {_value(group, 'profile_pred_accuracy')} | {_value(group, 'profile_kl')} | {_value(group, 'profile_target_confidence')} | {_value(group, 'profile_target_entropy')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    summary = summarize(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.markdown_out.write_text(render(summary), encoding="utf-8")
    print(f"ANALYZED_SAMPLES={summary['num_samples']}")
    print(f"PROFILE_AVAILABLE={summary['profile_available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
