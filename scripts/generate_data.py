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
from adaptive_learning.data.sqlite_store import write_dataset_to_sqlite


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
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=None,
        help="Optional SQLite file to mirror the generated dataset into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_cbse_math_dataset()
    write_dataset(dataset=dataset, output_dir=args.output_dir)
    if args.sqlite_path is not None:
        write_dataset_to_sqlite(dataset=dataset, sqlite_path=args.sqlite_path)

    question_sources = dataset.questions["source"].value_counts().to_dict()
    print(f"Wrote bootstrap dataset to {args.output_dir}")
    if args.sqlite_path is not None:
        print(f"Mirrored dataset into SQLite at {args.sqlite_path}")
    print(
        "Generated "
        f"{len(dataset.concepts)} concepts, "
        f"{len(dataset.questions)} questions, "
        f"{len(dataset.solutions)} solutions, and "
        f"{len(dataset.relationships)} relationships."
    )
    print(f"Question sources: {question_sources}")


if __name__ == "__main__":
    main()
