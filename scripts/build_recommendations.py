from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.data.generator import SUPPORTED_SUBJECTS
from adaptive_learning.recommendation.pipeline import run_recommendation_pipeline
from adaptive_learning.ui.data_access import default_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build adaptive practice recommendations from mastery artifacts.")
    parser.add_argument(
        "--subject",
        default="math",
        choices=SUPPORTED_SUBJECTS,
        help="Subject artifact set to use.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--mastery-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_paths(ROOT, subject=args.subject)
    data_dir = args.data_dir or paths.data_dir
    mastery_dir = args.mastery_dir or paths.mastery_dir
    output_dir = args.output_dir or paths.recommendation_dir
    result = run_recommendation_pipeline(
        data_dir=data_dir,
        mastery_dir=mastery_dir,
        output_dir=output_dir,
        top_k=args.top_k,
    )
    print(f"Subject: {args.subject}")
    print(f"Wrote recommendations to {result.output_dir}")
    print(f"Generated {len(result.recommendations)} ranked recommendations.")


if __name__ == "__main__":
    main()
