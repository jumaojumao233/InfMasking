"""Build a confusion matrix from an exported linear-probe prediction JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def build_confusion(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(y_true, y_pred):
        matrix[int(target)][int(prediction)] += 1
    return matrix


def make_summary(payload: dict) -> dict:
    y_true = [int(value) for value in payload["y_true"]]
    y_pred = [int(value) for value in payload["y_pred"]]
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths")
    num_classes = max(max(y_true, default=0), max(y_pred, default=0)) + 1
    matrix = build_confusion(y_true, y_pred, num_classes)
    rows = []
    for index, row in enumerate(matrix):
        support = sum(row)
        rows.append(
            {
                "class_id": index,
                "class_name": TEXTURES[index] if index < len(TEXTURES) else str(index),
                "support": support,
                "correct": row[index],
                "accuracy": row[index] / support if support else 0.0,
                "predicted_count": sum(matrix[target][index] for target in range(num_classes)),
            }
        )
    return {
        "num_samples": len(y_true),
        "num_classes": num_classes,
        "accuracy": sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0,
        "confusion_rows_true_columns_pred": matrix,
        "per_class": rows,
    }


def render(summary: dict) -> str:
    matrix = summary["confusion_rows_true_columns_pred"]
    labels = [row["class_name"] for row in summary["per_class"]]
    lines = [
        "# Checkpoint unique2 混淆矩阵",
        "",
        f"样本数：{summary['num_samples']}；整体 acc@1：{summary['accuracy']:.3f}。行是真实类别，列是预测类别。",
        "",
        "| true \\ pred | " + " | ".join(labels) + " | support | recall |",
        "|---|" + "---:|" * len(labels) + "---:|---:|",
    ]
    for row, info in zip(matrix, summary["per_class"]):
        lines.append(
            "| {name} | {values} | {support} | {accuracy:.3f} |".format(
                name=info["class_name"],
                values=" | ".join(str(value) for value in row),
                support=info["support"],
                accuracy=info["accuracy"],
            )
        )
    lines.extend(["", "| 类别 | 真实样本数 | 预测为本类的数量 | 类别准确率 |", "|---|---:|---:|---:|"])
    for info in summary["per_class"]:
        lines.append(
            f"| {info['class_name']} | {info['support']} | {info['predicted_count']} | {info['accuracy']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.prediction_json.read_text(encoding="utf-8"))
    summary = make_summary(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.markdown_out.write_text(render(summary), encoding="utf-8")
    print(f"ANALYZED {summary['num_samples']} samples")
    print(f"ACCURACY={summary['accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
