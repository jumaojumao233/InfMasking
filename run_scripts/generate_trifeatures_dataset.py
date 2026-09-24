"""Generate one reproducible Trifeatures train/test root for G0."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.trifeatures import Trifeatures  # noqa: E402


def count_images(root: Path, split: str) -> int:
    split_root = root / split
    if not split_root.exists():
        return 0
    return sum(1 for path in split_root.iterdir() if path.is_file() and path.suffix == ".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Output dataset root")
    parser.add_argument("--seed", type=int, required=True, help="Independent data-generation seed")
    parser.add_argument("--num-per-combination", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root)
    print(f"DATA_ROOT={root}")
    print(f"DATA_SEED={args.seed}")
    print(f"NUM_PER_COMBINATION={args.num_per_combination}")

    train = Trifeatures(
        str(root),
        split="train",
        num_per_combination=args.num_per_combination,
        seed=args.seed,
    )

    train_count = count_images(root, "train")
    test_count = count_images(root, "test")
    print(f"TRAIN_LEN={len(train)}")
    print(f"TRAIN_FILES={train_count}")
    print(f"TEST_FILES={test_count}")

    expected_train = 10 * 10 * 10 * args.num_per_combination * 0.8
    expected_test = 10 * 10 * 10 - 10 * 10 * 10 * 0.8
    if train_count != int(expected_train) or test_count != int(expected_test):
        raise RuntimeError(
            f"Incomplete Trifeatures root: train={train_count}, test={test_count}, "
            f"expected train={int(expected_train)}, test={int(expected_test)}"
        )

    print("DATASET_STATUS=complete")


if __name__ == "__main__":
    main()
