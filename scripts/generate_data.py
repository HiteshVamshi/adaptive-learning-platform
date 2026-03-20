from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.data.generator import generate_cbse_math_dataset
from adaptive_learning.data.io import write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured CBSE Class 10 mathematics bootstrap data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "bootstrap_data",
        help="Directory where generated data artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_cbse_math_dataset()
    write_dataset(dataset=dataset, output_dir=args.output_dir)
    print(f"Wrote bootstrap dataset to {args.output_dir}")
    print(
        "Generated "
        f"{len(dataset.concepts)} concepts, "
        f"{len(dataset.questions)} questions, "
        f"{len(dataset.solutions)} solutions, and "
        f"{len(dataset.relationships)} relationships."
    )


if __name__ == "__main__":
    main()
