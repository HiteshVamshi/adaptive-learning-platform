from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.content_generation.pipeline import load_generation_context
from adaptive_learning.content_generation.generator import build_content_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run content generation demo.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "bootstrap_data",
    )
    parser.add_argument(
        "--task",
        choices=["question", "explanation", "summary"],
        required=True,
    )
    parser.add_argument(
        "--concept-id",
        default="c_hcf_lcm",
    )
    parser.add_argument(
        "--difficulty",
        default="medium",
    )
    parser.add_argument(
        "--scope-type",
        choices=["concept", "chapter"],
        default="concept",
    )
    parser.add_argument(
        "--scope-id",
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


def main() -> None:
    args = parse_args()
    context = load_generation_context(data_dir=args.data_dir)
    generator = build_content_generator(
        context=context,
        backend=args.backend,
        model_name=args.model_name,
    )

    if args.task == "question":
        record, prompt = generator.generate_question(
            concept_id=args.concept_id,
            difficulty=args.difficulty,
        )
        payload = {"prompt_template": prompt, "output": record.to_dict()}
    elif args.task == "explanation":
        record, prompt = generator.generate_explanation(concept_id=args.concept_id)
        payload = {"prompt_template": prompt, "output": record.to_dict()}
    else:
        scope_id = args.scope_id or args.concept_id
        record, prompt = generator.generate_summary(scope_type=args.scope_type, scope_id=scope_id)
        payload = {"prompt_template": prompt, "output": record.to_dict()}

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
