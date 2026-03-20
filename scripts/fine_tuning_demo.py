from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.fine_tuning.adapter import DifficultyTunedGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show baseline vs adapted generation for difficulty control.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "bootstrap_data",
    )
    parser.add_argument(
        "--fine-tuning-dir",
        type=Path,
        default=ROOT / "artifacts" / "fine_tuning",
    )
    parser.add_argument(
        "--concept-id",
        default="c_hcf_lcm",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="hard",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuned_generator = DifficultyTunedGenerator.from_artifacts(
        data_dir=args.data_dir,
        fine_tuning_dir=args.fine_tuning_dir,
    )
    comparison = tuned_generator.compare_generation(
        concept_id=args.concept_id,
        target_difficulty=args.difficulty,
    )
    print(json.dumps(comparison.to_dict(), indent=2))


if __name__ == "__main__":
    main()
