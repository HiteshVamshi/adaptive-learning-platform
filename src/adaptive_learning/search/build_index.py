from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from networkx.readwrite import json_graph

from adaptive_learning.search.hybrid_search import build_hybrid_search_engine, build_search_documents


@dataclass(frozen=True)
class SearchIndexBuildResult:
    documents: pd.DataFrame
    embedding_backend: str
    embedding_model_name: str
    output_dir: Path


def build_search_index(
    *,
    data_dir: Path,
    output_dir: Path,
    embedding_backend: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
) -> SearchIndexBuildResult:
    concepts = pd.read_csv(data_dir / "concepts.csv")
    questions = pd.read_csv(data_dir / "questions.csv")
    solutions = pd.read_csv(data_dir / "solutions.csv")

    with (data_dir / "concept_graph.json").open("r", encoding="utf-8") as file_obj:
        concept_graph = json_graph.node_link_graph(json.load(file_obj), multigraph=True)

    documents = build_search_documents(
        questions=questions,
        solutions=solutions,
        concepts=concepts,
    )

    engine = build_hybrid_search_engine(
        documents=documents,
        concepts=concepts,
        concept_graph=concept_graph,
        embedding_backend=embedding_backend,
        model_name=model_name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    engine.vector_store.save(output_dir)

    build_summary = {
        "document_count": int(len(documents)),
        "embedding_backend": engine.vector_store.embedder.backend_name,
        "embedding_model_name": getattr(engine.vector_store.embedder, "model_name", None),
        "chapters": sorted(documents["chapter_name"].unique().tolist()),
        "difficulties": sorted(documents["difficulty"].unique().tolist()),
    }
    with (output_dir / "index_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(build_summary, file_obj, indent=2)

    return SearchIndexBuildResult(
        documents=documents,
        embedding_backend=engine.vector_store.embedder.backend_name,
        embedding_model_name=getattr(engine.vector_store.embedder, "model_name", model_name),
        output_dir=output_dir,
    )
