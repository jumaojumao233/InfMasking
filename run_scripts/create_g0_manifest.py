"""Create a stable file-hash manifest and summary for a G0 Trifeatures root."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest).resolve()
    summary_path = Path(args.summary).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())

    lines = []
    split_counts = {"train": 0, "test": 0}
    split_bytes = {"train": 0, "test": 0}
    for path in files:
        relative = path.relative_to(root).as_posix()
        split = relative.split("/", 1)[0]
        if split in split_counts:
            split_counts[split] += 1
            split_bytes[split] += path.stat().st_size
        lines.append(f"{sha256_file(path)}  {relative}")

    manifest_text = "\n".join(lines) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    summary = {
        "root": str(root),
        "data_seed": args.data_seed,
        "file_count": len(files),
        "split_counts": split_counts,
        "split_bytes": split_bytes,
        "total_bytes": sum(split_bytes.values()),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
