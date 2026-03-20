from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adaptive_learning.rag.chunking import build_rag_chunks
from adaptive_learning.rag.pipeline import build_rag_vector_store


@dataclass(frozen=True)
class RAGIndexBuildResult:
    chunk_count: int
    output_dir: Path
    embedding_backend: str
    embedding_model_name: str


def build_rag_index(
    *,
    data_dir: Path,
    output_dir: Path,
    embedding_backend: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
) -> RAGIndexBuildResult:
    concepts = pd.read_csv(data_dir / "concepts.csv")
    questions = pd.read_csv(data_dir / "questions.csv")
    solutions = pd.read_csv(data_dir / "solutions.csv")
    theory_content_path = data_dir / "theory_content.csv"
    theory_content = pd.read_csv(theory_content_path) if theory_content_path.exists() else None

    chunks = build_rag_chunks(
        concepts=concepts,
        questions=questions,
        solutions=solutions,
        theory_content=theory_content,
    )
    vector_store = build_rag_vector_store(
        chunks=chunks,
        embedding_backend=embedding_backend,
        model_name=model_name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save(output_dir)

    summary = {
        "chunk_count": int(len(chunks)),
        "embedding_backend": vector_store.embedder.backend_name,
        "embedding_model_name": getattr(vector_store.embedder, "model_name", None),
        "chunk_types": chunks["chunk_type"].value_counts().to_dict(),
    }
    with (output_dir / "rag_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return RAGIndexBuildResult(
        chunk_count=int(len(chunks)),
        output_dir=output_dir,
        embedding_backend=vector_store.embedder.backend_name,
        embedding_model_name=getattr(vector_store.embedder, "model_name", model_name),
    )
