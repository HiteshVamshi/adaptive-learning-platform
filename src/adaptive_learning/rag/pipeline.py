from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from networkx.readwrite import json_graph

from adaptive_learning.rag.generator import GeneratedAnswer, build_answer_generator
from adaptive_learning.search.query_understanding import QueryIntent, QueryUnderstandingEngine
from adaptive_learning.search.vector_store import FaissVectorStore, build_embedder


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    chunk_type: str
    source_id: str
    chapter_name: str
    concept_name: str
    difficulty: str
    title: str
    content: str
    score: float


@dataclass(frozen=True)
class RAGResponse:
    query: str
    intent: QueryIntent
    answer: GeneratedAnswer
    retrieved_chunks: list[RetrievedChunk]


class RAGEngine:
    def __init__(
        self,
        *,
        chunks: pd.DataFrame,
        vector_store: FaissVectorStore,
        query_engine: QueryUnderstandingEngine,
        generator_backend: str = "auto",
        generator_model_name: str = "google/flan-t5-base",
    ) -> None:
        self.chunks = chunks.copy()
        self.chunks_by_id = self.chunks.set_index("chunk_id")
        self.vector_store = vector_store
        self.query_engine = query_engine
        self.answer_generator = build_answer_generator(
            generator_backend=generator_backend,
            model_name=generator_model_name,
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        data_dir: Path,
        index_dir: Path,
        embedding_backend: str | None = None,
        embedding_model_name: str | None = None,
        generator_backend: str = "auto",
        generator_model_name: str = "google/flan-t5-base",
    ) -> "RAGEngine":
        concepts = pd.read_csv(data_dir / "concepts.csv")
        with (data_dir / "concept_graph.json").open("r", encoding="utf-8") as file_obj:
            concept_graph = json_graph.node_link_graph(json.load(file_obj), multigraph=True)

        vector_store = FaissVectorStore.load(
            input_dir=index_dir,
            embedding_backend=embedding_backend,
            model_name=embedding_model_name,
        )
        query_engine = QueryUnderstandingEngine(concepts=concepts, concept_graph=concept_graph)

        return cls(
            chunks=vector_store.documents,
            vector_store=vector_store,
            query_engine=query_engine,
            generator_backend=generator_backend,
            generator_model_name=generator_model_name,
        )

    def answer_question(self, query: str, top_k: int = 4) -> RAGResponse:
        intent = self.query_engine.understand(query)
        expanded_query = self._expanded_query(intent)
        vector_hits = self.vector_store.search(expanded_query, top_k=top_k)
        vector_hits = self._enrich_with_linked_solutions(vector_hits)

        retrieved_chunks: list[RetrievedChunk] = []
        for hit in vector_hits[:top_k]:
            row = self.chunks_by_id.loc[hit.doc_id]
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=str(hit.doc_id),
                    chunk_type=str(row["chunk_type"]),
                    source_id=str(row["source_id"]),
                    chapter_name=str(row["chapter_name"]),
                    concept_name=str(row["concept_name"]),
                    difficulty=str(row["difficulty"]),
                    title=str(row["title"]),
                    content=str(row["content"]),
                    score=float(hit.score),
                )
            )

        answer = self.answer_generator.generate(
            query=query,
            retrieved_chunks=[asdict(chunk) for chunk in retrieved_chunks],
        )
        return RAGResponse(
            query=query,
            intent=intent,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
        )

    def _enrich_with_linked_solutions(self, vector_hits: list) -> list:
        enriched_hits = list(vector_hits)
        seen_doc_ids = {hit.doc_id for hit in vector_hits}
        for hit in vector_hits:
            if not str(hit.doc_id).startswith("question::"):
                continue
            question_id = str(hit.doc_id).split("question::", 1)[1]
            solution_chunk_id = f"solution::{question_id}"
            if solution_chunk_id in self.chunks_by_id.index and solution_chunk_id not in seen_doc_ids:
                enriched_hits.append(
                    type(hit)(doc_id=solution_chunk_id, score=hit.score * 0.98, rank=hit.rank + 1)
                )
                seen_doc_ids.add(solution_chunk_id)
        return sorted(enriched_hits, key=lambda item: item.score, reverse=True)

    def _expanded_query(self, intent: QueryIntent) -> str:
        segments = [intent.original_query, *intent.expanded_terms]
        return " ".join(segment for segment in segments if segment)


def build_rag_vector_store(
    *,
    chunks: pd.DataFrame,
    embedding_backend: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
) -> FaissVectorStore:
    vector_ready_chunks = chunks.rename(columns={"retrieval_text": "search_text"}).copy()
    vector_ready_chunks["doc_id"] = vector_ready_chunks["chunk_id"]
    embedder = build_embedder(
        embedding_backend=embedding_backend,
        model_name=model_name,
    )
    return FaissVectorStore.build(documents=vector_ready_chunks, embedder=embedder)
