"""Parse G0 linear-probe logs and summarize unique2 class-level changes.

This is a CPU-only log analysis utility. It does not load checkpoints or
instantiate a model. The texture order is defined by dataset/trifeatures.py:
solid, stripes, grid, hexgrid, dots, noise, triangles, zigzags, rain, pluses.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
PROBES = ["share", "unique1", "unique2", "synergy"]
METRIC_RE = re.compile(
    r"Test acc@1/acc@5/acc_per_class/roc_auc:\s*"
    r"([0-9.]+)/([0-9.]+)/\[([^]]+)\]/([0-9.]+)"
)
SEED_RE = re.compile(r"_s(\d+)_")


def parse_file(path: Path) -> dict:
    method = "unigir" if "_unigir" in path.stem.lower() else "baseline"
    seed_match = SEED_RE.search(path.stem)
    if seed_match is None:
        raise ValueError(f"cannot infer seed from {path.name}")
    seed = int(seed_match.group(1))
    metrics = []
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        match = METRIC_RE.search(line)
        if match is None:
            continue
        classes = [float(value.strip()) for value in match.group(3).split(",")]
        metrics.append(
            {
                "acc1": float(match.group(1)),
                "acc5": float(match.group(2)),
                "acc_per_class": classes,
                "roc_auc": float(match.group(4)),
            }
        )
    if len(metrics) != len(PROBES):
        raise ValueError(f"expected four probes in {path.name}, got {len(metrics)}")
    expected_class_counts = [len(TEXTURES), len(TEXTURES), len(TEXTURES), 2]
    for probe, metric, expected_count in zip(PROBES, metrics, expected_class_counts):
        if len(metric["acc_per_class"]) != expected_count:
            raise ValueError(
                f"expected {expected_count} class accuracies for {probe} in {path.name}, "
                f"got {len(metric['acc_per_class'])}"
            )
    return {
        "file": path.name,
        "method": method,
        "seed": seed,
        "probes": dict(zip(PROBES, metrics)),
    }


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def fmt(value: float) -> str:
    return f"{value:.3f}"


def build_summary(records: list[dict]) -> dict:
    by_key = {(record["seed"], record["method"]): record for record in records}
    paired = []
    for seed in sorted({record["seed"] for record in records}):
        baseline = by_key[(seed, "baseline")]
        unigir = by_key[(seed, "unigir")]
        unique2_delta = [
            b - a
            for a, b in zip(
                baseline["probes"]["unique2"]["acc_per_class"],
                unigir["probes"]["unique2"]["acc_per_class"],
            )
        ]
        paired.append(
            {
                "seed": seed,
                "baseline": baseline["probes"]["unique2"],
                "unigir": unigir["probes"]["unique2"],
                "unigir_minus_baseline": {
                    "acc1": unigir["probes"]["unique2"]["acc1"]
                    - baseline["probes"]["unique2"]["acc1"],
                    "roc_auc": unigir["probes"]["unique2"]["roc_auc"]
                    - baseline["probes"]["unique2"]["roc_auc"],
                    "acc_per_class": unique2_delta,
                },
            }
        )
    class_deltas = [item["unigir_minus_baseline"]["acc_per_class"] for item in paired]
    mean_delta = [mean(values) for values in zip(*class_deltas)]
    std_delta = [sample_std(list(values)) for values in zip(*class_deltas)]
    return {
        "texture_order": TEXTURES,
        "records": records,
        "unique2_pairs": paired,
        "unique2_class_delta_mean": mean_delta,
        "unique2_class_delta_sample_std": std_delta,
    }


def markdown(summary: dict) -> str:
    lines = [
        "# G0 unique2 类别级日志分析",
        "",
        "输入是六个 G0 stdout 日志中的 `acc_per_class`。类别顺序依据 `dataset/trifeatures.py` 的 `BASE_TEXTURES`，未从 checkpoint 重新计算预测，也没有混淆矩阵。",
        "",
        "## 每个 seed 的 unique2 类别准确率",
        "",
        "| seed | texture | Baseline | UniGIR | UniGIR-Baseline |",
        "|---:|---|---:|---:|---:|",
    ]
    for pair in summary["unique2_pairs"]:
        for index, texture in enumerate(TEXTURES):
            baseline = pair["baseline"]["acc_per_class"][index]
            unigir = pair["unigir"]["acc_per_class"][index]
            delta = pair["unigir_minus_baseline"]["acc_per_class"][index]
            lines.append(
                f"| {pair['seed']} | {texture} | {fmt(baseline)} | {fmt(unigir)} | {fmt(delta)} |"
            )
    lines.extend(
        [
            "",
            "## 三个 seed 的变化",
            "",
            "| texture | 平均变化 | 样本标准差 | 3 个 seed 中下降次数 |",
            "|---|---:|---:|---:|",
        ]
    )
    for index, texture in enumerate(TEXTURES):
        values = [
            pair["unigir_minus_baseline"]["acc_per_class"][index]
            for pair in summary["unique2_pairs"]
        ]
        lines.append(
            f"| {texture} | {fmt(mean(values))} | {fmt(sample_std(values))} | {sum(value < 0 for value in values)} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "unique2 对应代码中的 `unique_attr=texture`，所以类别编号可以映射到纹理名称。日志没有保存每个样本的预测标签，当前结果只能定位到类别准确率的变化，不能声称已经得到混淆矩阵或具体错分方向。下一步若要判断错分对，需在 winpc 上增加一次只读 checkpoint evaluation，导出 `y_true`、`y_pred` 和 logits；该任务应使用独立日志、独立输出目录和同一封存 G0 数据。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.log_dir.glob("winpc_g4_g0_trifeatures_s*.stdout.log"))
    records = []
    skipped = []
    for path in paths:
        try:
            records.append(parse_file(path))
        except ValueError as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
    if len(records) != 6:
        raise SystemExit(f"expected six complete G0 stdout logs, found {len(records)}; skipped={skipped}")
    summary = build_summary(records)
    summary["skipped_logs"] = skipped
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.markdown_out.write_text(markdown(summary), encoding="utf-8")
    print(f"ANALYZED {len(records)} logs")
    print(f"JSON={args.json_out}")
    print(f"MARKDOWN={args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
