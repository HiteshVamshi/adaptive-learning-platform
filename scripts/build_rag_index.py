from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.rag.build_index import build_rag_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RAG chunk index.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "bootstrap_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "rag_index",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence_transformer", "hash"],
        default="auto",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_rag_index(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        embedding_backend=args.embedding_backend,
        model_name=args.embedding_model,
    )
    print(f"Wrote RAG index to {result.output_dir}")
    print(
        "Indexed "
        f"{result.chunk_count} chunks using backend={result.embedding_backend} "
        f"model={result.embedding_model_name}"
    )


if __name__ == "__main__":
    main()
