from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.fine_tuning.pipeline import run_fine_tuning_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight difficulty-calibration adapter.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "bootstrap_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "fine_tuning",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_fine_tuning_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        concept_pairs=[
            ("c_hcf_lcm", "hard"),
            ("c_trigonometric_ratios", "medium"),
            ("c_area_of_triangle", "easy"),
        ],
        random_seed=args.random_seed,
    )
    print(f"Wrote fine-tuning artifacts to {result.output_dir}")
    print(
        f"Trained on {len(result.dataset)} examples with held-out accuracy "
        f"{result.metrics['accuracy']:.3f} and generated {len(result.comparisons)} adaptation comparisons."
    )


if __name__ == "__main__":
    main()
