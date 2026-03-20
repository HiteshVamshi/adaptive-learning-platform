from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.content_generation.pipeline import build_content_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generated content samples.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "bootstrap_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "generated_content",
    )
    parser.add_argument(
        "--concept-ids",
        nargs="+",
        default=["c_hcf_lcm", "c_trigonometric_ratios", "c_area_of_triangle"],
    )
    parser.add_argument(
        "--backend",
        choices=["grounded", "transformers", "auto"],
        default="grounded",
    )
    parser.add_argument(
        "--model-name",
        default="google/flan-t5-base",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_content_bundle(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        concept_ids=args.concept_ids,
        backend=args.backend,
        model_name=args.model_name,
    )
    print(f"Wrote generated content bundle to {result.output_dir}")
    print(
        f"Generated {len(result.questions)} questions, {len(result.explanations)} explanations, "
        f"and {len(result.summaries)} summaries."
    )


if __name__ == "__main__":
    main()
