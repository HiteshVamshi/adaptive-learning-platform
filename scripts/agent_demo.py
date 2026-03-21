from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.agents.io import write_agent_trace
from adaptive_learning.agents.pipeline import build_agent_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a modular agent demo.")
    parser.add_argument(
        "--agent",
        choices=["tutor", "practice", "query", "learn"],
        required=True,
    )
    parser.add_argument(
        "--query",
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "bootstrap_data",
    )
    parser.add_argument(
        "--search-index-dir",
        type=Path,
        default=ROOT / "artifacts" / "search_index",
    )
    parser.add_argument(
        "--rag-index-dir",
        type=Path,
        default=ROOT / "artifacts" / "rag_index",
    )
    parser.add_argument(
        "--mastery-dir",
        type=Path,
        default=ROOT / "artifacts" / "mastery",
    )
    parser.add_argument(
        "--recommendation-dir",
        type=Path,
        default=ROOT / "artifacts" / "recommendations",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = build_agent_suite(
        data_dir=args.data_dir,
        search_index_dir=args.search_index_dir,
        rag_index_dir=args.rag_index_dir,
        mastery_dir=args.mastery_dir,
        recommendation_dir=args.recommendation_dir,
    )
    agent = suite.get(args.agent)
    response = agent.respond(args.query)

    if args.trace_out:
        write_agent_trace(output_path=args.trace_out, response=response)

    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    main()
