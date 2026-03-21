from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.data.generator import SUPPORTED_SUBJECTS, generate_subject_dataset
from adaptive_learning.data.io import write_dataset
from adaptive_learning.data.sqlite_store import write_dataset_to_sqlite
from adaptive_learning.learn.build_index import write_learn_index_for_subject
from adaptive_learning.ui.data_access import default_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured CBSE Class 10 subject bootstrap data."
    )
    parser.add_argument(
        "--subject",
        default="math",
        choices=SUPPORTED_SUBJECTS,
        help="Subject to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where generated data artifacts will be written. Defaults to artifacts/subjects/<subject>/bootstrap_data.",
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
    paths = default_paths(ROOT, subject=args.subject)
    output_dir = args.output_dir or paths.data_dir
    sqlite_path = args.sqlite_path or (output_dir / "adaptive_learning.db")

    dataset = generate_subject_dataset(args.subject)
    write_dataset(dataset=dataset, output_dir=output_dir)
    write_dataset_to_sqlite(dataset=dataset, sqlite_path=sqlite_path)

    learn_out = write_learn_index_for_subject(artifacts_dir=paths.artifacts_dir, subject=args.subject)
    question_sources = dataset.questions["source"].value_counts().to_dict()
    print(f"Subject: {args.subject}")
    print(f"Wrote bootstrap dataset to {output_dir}")
    print(f"Mirrored dataset into SQLite at {sqlite_path}")
    print(f"Wrote learn catalog to {learn_out}")
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
