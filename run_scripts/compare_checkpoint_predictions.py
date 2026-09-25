"""Compare two paired checkpoint prediction exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_checkpoint_predictions import TEXTURES, build_confusion, make_summary


def compare(baseline: dict, unigir: dict) -> dict:
    if baseline["y_true"] != unigir["y_true"]:
        raise ValueError("paired evaluations do not have identical y_true order")
    base = make_summary(baseline)
    new = make_summary(unigir)
    recall_delta = [
        b["accuracy"] - a["accuracy"]
        for a, b in zip(base["per_class"], new["per_class"])
    ]
    base_matrix = base["confusion_rows_true_columns_pred"]
    new_matrix = new["confusion_rows_true_columns_pred"]
    matrix_delta = [
        [new_value - base_value for base_value, new_value in zip(base_row, new_row)]
        for base_row, new_row in zip(base_matrix, new_matrix)
    ]
    return {
        "baseline": base,
        "unigir": new,
        "accuracy_delta": new["accuracy"] - base["accuracy"],
        "recall_delta": recall_delta,
        "confusion_delta": matrix_delta,
    }


def render(summary: dict, label: str) -> str:
    lines = [
        f"# {label} checkpoint 配对评测",
        "",
        f"Baseline 整体 acc@1：{summary['baseline']['accuracy']:.3f}；UniGIR：{summary['unigir']['accuracy']:.3f}；差值：{summary['accuracy_delta']:+.3f}。两次评测使用同一 `y_true` 顺序和同一批封存 G0 数据。",
        "",
        "| texture | Baseline recall | UniGIR recall | UniGIR - Baseline |",
        "|---|---:|---:|---:|",
    ]
    for index, texture in enumerate(TEXTURES):
        base_recall = summary["baseline"]["per_class"][index]["accuracy"]
        unigir_recall = summary["unigir"]["per_class"][index]["accuracy"]
        lines.append(f"| {texture} | {base_recall:.3f} | {unigir_recall:.3f} | {summary['recall_delta'][index]:+.3f} |")
    changes = []
    matrix = summary["confusion_delta"]
    for true_index, row in enumerate(matrix):
        for pred_index, value in enumerate(row):
            if true_index != pred_index and value != 0:
                changes.append((abs(value), value, TEXTURES[true_index], TEXTURES[pred_index]))
    changes.sort(reverse=True)
    lines.extend(["", "## 变化最大的非对角混淆项", "", "| 真实类别 | 预测类别 | UniGIR - Baseline 样本数 |", "|---|---|---:|"])
    for _, value, true_name, pred_name in changes[:12]:
        lines.append(f"| {true_name} | {pred_name} | {value:+d} |")
    lines.extend([
        "",
        "评测使用与 G0 训练末 probing 不同的独立进程。由于当前 test transform 包含随机裁剪，绝对 acc@1 与训练日志不能直接合并；本表只用于同 seed、同命令顺序下的 Baseline/UniGIR 配对诊断。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--unigir-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--label", default="G0 unique2", help="写入 Markdown 标题的评测标签")
    args = parser.parse_args()
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    unigir = json.loads(args.unigir_json.read_text(encoding="utf-8"))
    summary = compare(baseline, unigir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.markdown_out.write_text(render(summary, args.label), encoding="utf-8")
    print(f"ACCURACY_DELTA={summary['accuracy_delta']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
