from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect mastery outputs for a simulated learner.")
    parser.add_argument(
        "--mastery-dir",
        type=Path,
        default=ROOT / "artifacts" / "mastery",
    )
    parser.add_argument(
        "--concept-id",
        default=None,
        help="Optional concept id to inspect history for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = pd.read_csv(args.mastery_dir / "mastery_snapshot.csv")
    history = pd.read_csv(args.mastery_dir / "mastery_history.csv")
    attempts = pd.read_csv(args.mastery_dir / "attempts.csv")

    weakest = snapshot.sort_values(by="graph_adjusted_mastery", ascending=True).head(args.top_k)
    strongest = snapshot.sort_values(by="graph_adjusted_mastery", ascending=False).head(args.top_k)

    payload = {
        "attempt_count": int(len(attempts)),
        "weakest_concepts": weakest[
            [
                "concept_id",
                "concept_name",
                "graph_adjusted_mastery",
                "mastery_band",
                "explanation",
            ]
        ].to_dict(orient="records"),
        "strongest_concepts": strongest[
            [
                "concept_id",
                "concept_name",
                "graph_adjusted_mastery",
                "mastery_band",
                "explanation",
            ]
        ].to_dict(orient="records"),
    }

    if args.concept_id:
        concept_history = history[history["concept_id"] == args.concept_id].tail(10)
        payload["concept_history"] = concept_history.to_dict(orient="records")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
