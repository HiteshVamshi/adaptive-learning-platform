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
    parser = argparse.ArgumentParser(description="Inspect adaptive recommendation outputs.")
    parser.add_argument(
        "--recommendation-dir",
        type=Path,
        default=ROOT / "artifacts" / "recommendations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommendations = pd.read_csv(args.recommendation_dir / "recommendations.csv")
    top_rows = recommendations.head(args.top_k)
    payload = {
        "recommendation_count": int(len(recommendations)),
        "top_recommendations": top_rows.to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
