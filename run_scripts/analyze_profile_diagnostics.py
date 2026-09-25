"""Extract aggregated UniGIR profile metrics from a read-only test log."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def parse_test_results(stdout_text: str) -> dict[str, float]:
    matches = re.findall(r"Test results:\s*(\[.*?\])", stdout_text, flags=re.DOTALL)
    if not matches:
        raise ValueError("could not find a Test results line")
    parsed = ast.literal_eval(matches[-1])
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
        raise ValueError("Test results does not contain a metric dictionary")
    metrics = {}
    for key, value in parsed[0].items():
        if key.startswith("test_") and isinstance(value, (int, float)):
            metrics[key] = float(value)
    if not metrics:
        raise ValueError("Test results contains no numeric test_ metrics")
    return metrics


def render_markdown(run_name: str, metrics: dict[str, float]) -> str:
    lines = [
        f"# {run_name} profile 只读诊断",
        "",
        "评测使用固定 `Resize(256) + CenterCrop(224)`，运行 `mode=test`，没有调用 `trainer.fit`。",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key, value in sorted(metrics.items()):
        lines.append(f"| `{key}` | {value:.6f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    stdout_text = args.stdout_log.read_bytes().decode("utf-8", errors="replace")
    metrics = parse_test_results(stdout_text)
    payload = {"run_name": args.run_name, "metrics": metrics}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.markdown_out.write_text(render_markdown(args.run_name, metrics), encoding="utf-8")
    print(f"PROFILE_METRICS={len(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
