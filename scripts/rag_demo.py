from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.rag.pipeline import RAGEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grounded RAG QA demo.")
    parser.add_argument("query", help="Question to answer from the RAG corpus.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "bootstrap_data",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "artifacts" / "rag_index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
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
    parser.add_argument(
        "--generator-backend",
        choices=["auto", "grounded", "transformers"],
        default="grounded",
    )
    parser.add_argument(
        "--generator-model",
        default="google/flan-t5-base",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = RAGEngine.from_artifacts(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        embedding_backend=args.embedding_backend,
        embedding_model_name=args.embedding_model,
        generator_backend=args.generator_backend,
        generator_model_name=args.generator_model,
    )
    response = engine.answer_question(args.query, top_k=args.top_k)

    payload = {
        "query": response.query,
        "intent": {
            "requested_difficulty": response.intent.requested_difficulty,
            "detected_concepts": response.intent.detected_concept_ids,
            "expanded_concepts": response.intent.expanded_concept_ids,
        },
        "answer": response.answer.answer,
        "generator_backend": response.answer.backend,
        "prompt": response.answer.prompt,
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type,
                "source_id": chunk.source_id,
                "chapter_name": chunk.chapter_name,
                "concept_name": chunk.concept_name,
                "difficulty": chunk.difficulty,
                "title": chunk.title,
                "content": chunk.content,
                "score": round(chunk.score, 4),
            }
            for chunk in response.retrieved_chunks
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
