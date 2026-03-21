from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.data.generator import SUPPORTED_SUBJECTS
from adaptive_learning.search.build_index import build_search_index
from adaptive_learning.ui.data_access import default_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the hybrid search index.")
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
        help="Directory containing generated subject data artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where search index artifacts will be written.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence_transformer", "hash"],
        default="auto",
        help="Embedding backend used for vector search.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name when that backend is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_paths(ROOT, subject=args.subject)
    data_dir = args.data_dir or paths.data_dir
    output_dir = args.output_dir or paths.search_index_dir
    result = build_search_index(
        data_dir=data_dir,
        output_dir=output_dir,
        embedding_backend=args.embedding_backend,
        model_name=args.embedding_model,
    )
    print(f"Subject: {args.subject}")
    print(f"Wrote search index to {result.output_dir}")
    print(
        "Indexed "
        f"{len(result.documents)} documents using backend={result.embedding_backend} "
        f"model={result.embedding_model_name}"
    )


if __name__ == "__main__":
    main()
