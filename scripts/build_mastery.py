from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.data.generator import SUPPORTED_SUBJECTS
from adaptive_learning.mastery.pipeline import run_mastery_pipeline
from adaptive_learning.ui.data_access import default_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate student attempts and build mastery artifacts.")
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
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--user-id",
        default="student_cbse_01",
    )
    parser.add_argument(
        "--num-attempts",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_paths(ROOT, subject=args.subject)
    data_dir = args.data_dir or paths.data_dir
    output_dir = args.output_dir or paths.mastery_dir
    result = run_mastery_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        user_id=args.user_id,
        num_attempts=args.num_attempts,
        random_seed=args.random_seed,
    )
    print(f"Subject: {args.subject}")
    print(f"Wrote mastery artifacts to {result.output_dir}")
    print(
        f"Simulated {len(result.attempts)} attempts for {result.student_profile.display_name} "
        f"and generated {len(result.snapshot)} concept mastery rows."
    )


if __name__ == "__main__":
    main()
