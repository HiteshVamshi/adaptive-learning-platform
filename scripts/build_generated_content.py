from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.content_generation.pipeline import build_content_bundle
from adaptive_learning.data.generator import SUPPORTED_SUBJECTS
from adaptive_learning.ui.data_access import default_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generated content samples.")
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
        "--concept-ids",
        nargs="+",
        default=None,
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


def _default_concept_ids(data_dir: Path) -> list[str]:
    concepts = pd.read_csv(data_dir / "concepts.csv")
    concept_rows = concepts[concepts["node_type"] == "concept"].copy()
    preferred_ids = [
        "c_hcf_lcm",
        "c_trigonometric_ratios",
        "c_classical_probability",
        "c_acids_bases_ph",
        "c_lenses_refraction",
        "c_ohms_law_resistance",
    ]
    available = concept_rows["concept_id"].astype(str).tolist()
    selected = [concept_id for concept_id in preferred_ids if concept_id in available]
    if len(selected) >= 3:
        return selected[:3]
    return concept_rows.sort_values(by=["chapter_name", "order_index"]).head(3)["concept_id"].astype(str).tolist()


def main() -> None:
    args = parse_args()
    paths = default_paths(ROOT, subject=args.subject)
    data_dir = args.data_dir or paths.data_dir
    output_dir = args.output_dir or paths.generated_content_dir
    concept_ids = args.concept_ids or _default_concept_ids(data_dir)
    result = build_content_bundle(
        data_dir=data_dir,
        output_dir=output_dir,
        concept_ids=concept_ids,
        backend=args.backend,
        model_name=args.model_name,
    )
    print(f"Subject: {args.subject}")
    print(f"Wrote generated content bundle to {result.output_dir}")
    print(
        f"Generated {len(result.questions)} questions, {len(result.explanations)} explanations, "
        f"and {len(result.summaries)} summaries."
    )


if __name__ == "__main__":
    main()
