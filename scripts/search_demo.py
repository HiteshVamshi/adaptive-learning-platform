from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.search.hybrid_search import HybridSearchEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hybrid retrieval query demo.")
    parser.add_argument("query", help="User query to retrieve against the question corpus.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "bootstrap_data",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "artifacts" / "search_index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence_transformer", "hash"],
        default=None,
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = HybridSearchEngine.from_artifacts(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        embedding_backend=args.embedding_backend,
        model_name=args.embedding_model,
    )
    intent, results = engine.search(args.query, top_k=args.top_k)

    payload = {
        "query": args.query,
        "intent": {
            "requested_difficulty": intent.requested_difficulty,
            "detected_concepts": intent.detected_concept_ids,
            "expanded_concepts": intent.expanded_concept_ids,
            "expanded_terms": intent.expanded_terms,
        },
        "results": [
            {
                "doc_id": result.doc_id,
                "question_id": result.question_id,
                "chapter_name": result.chapter_name,
                "concept_name": result.concept_name,
                "difficulty": result.difficulty,
                "prompt": result.prompt,
                "final_answer": result.final_answer,
                "bm25_score": round(result.bm25_score, 4),
                "vector_score": round(result.vector_score, 4),
                "hybrid_score": round(result.hybrid_score, 4),
                "exact_concept_match": result.exact_concept_match,
                "concept_overlap": result.concept_overlap,
                "difficulty_match": result.difficulty_match,
            }
            for result in results
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
