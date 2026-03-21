from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.data.generator import SUPPORTED_SUBJECTS
from adaptive_learning.fine_tuning.pipeline import run_fine_tuning_pipeline
from adaptive_learning.ui.data_access import default_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight difficulty-calibration adapter.")
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
        "--random-seed",
        type=int,
        default=7,
    )
    return parser.parse_args()


def _default_pairs(data_dir: Path) -> list[tuple[str, str]]:
    concepts = pd.read_csv(data_dir / "concepts.csv")
    concept_rows = concepts[concepts["node_type"] == "concept"].copy()
    if concept_rows.empty:
        return []

    difficulty_map = {"easy": None, "medium": None, "hard": None}
    for row in concept_rows.sort_values(by=["chapter_name", "order_index"]).to_dict(orient="records"):
        band = str(row.get("difficulty_band") or "medium")
        if band == "core":
            band = "medium"
        if band in difficulty_map and difficulty_map[band] is None:
            difficulty_map[band] = str(row["concept_id"])

    pairs = []
    seen = set()
    for difficulty in ["hard", "easy", "medium"]:
        concept_id = difficulty_map.get(difficulty)
        if concept_id and concept_id not in seen:
            pairs.append((concept_id, difficulty))
            seen.add(concept_id)

    if pairs:
        return pairs

    first_three = concept_rows.head(3)["concept_id"].astype(str).tolist()
    fallback_difficulties = ["easy", "medium", "hard"]
    return list(zip(first_three, fallback_difficulties[: len(first_three)]))


def main() -> None:
    args = parse_args()
    paths = default_paths(ROOT, subject=args.subject)
    data_dir = args.data_dir or paths.data_dir
    output_dir = args.output_dir or paths.fine_tuning_dir
    result = run_fine_tuning_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        concept_pairs=_default_pairs(data_dir),
        random_seed=args.random_seed,
    )
    print(f"Subject: {args.subject}")
    print(f"Wrote fine-tuning artifacts to {result.output_dir}")
    print(
        f"Trained on {len(result.dataset)} examples with held-out accuracy "
        f"{result.metrics['accuracy']:.3f} and generated {len(result.comparisons)} adaptation comparisons."
    )


if __name__ == "__main__":
    main()
