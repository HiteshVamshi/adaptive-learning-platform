from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adaptive_learning.data.curriculum_artifacts import load_concepts_and_graph
from adaptive_learning.search.bm25 import BM25Retriever
from adaptive_learning.search.query_understanding import QueryIntent, QueryUnderstandingEngine
from adaptive_learning.search.vector_store import FaissVectorStore, build_embedder


@dataclass(frozen=True)
class HybridRankingWeights:
    """Keyword vs vector vs intent signals for hybrid_score (sum need not be 1)."""

    bm25: float = 0.36
    vector: float = 0.28
    exact_concept: float = 0.28
    concept_overlap: float = 0.05
    difficulty_match: float = 0.03


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    question_id: str
    chapter_name: str
    concept_name: str
    difficulty: str
    prompt: str
    final_answer: str
    bm25_score: float
    vector_score: float
    hybrid_score: float
    exact_concept_match: bool
    concept_overlap: bool
    difficulty_match: bool


def build_search_documents(
    *,
    questions: pd.DataFrame,
    solutions: pd.DataFrame,
    concepts: pd.DataFrame,
) -> pd.DataFrame:
    merged = questions.merge(solutions, on="question_id", how="left")
    merged = merged.merge(
        concepts[
            [
                "concept_id",
                "description",
                "learning_objective",
                "chapter_name",
            ]
        ],
        on=["concept_id", "chapter_name"],
        how="left",
        suffixes=("", "_concept"),
    )

    documents = pd.DataFrame(
        {
            "doc_id": merged["question_id"].map(lambda value: f"question::{value}"),
            "question_id": merged["question_id"],
            "chapter_id": merged["chapter_id"],
            "chapter_name": merged["chapter_name"],
            "concept_id": merged["concept_id"],
            "concept_name": merged["concept_name"],
            "difficulty": merged["difficulty"],
            "question_type": merged["question_type"],
            "prompt": merged["prompt"],
            "final_answer": merged["final_answer"],
            "worked_solution": merged["worked_solution"],
            "tags": merged["tags"],
        }
    )
    documents["search_text"] = (
        documents["chapter_name"].fillna("")
        + " "
        + documents["concept_name"].fillna("")
        + " "
        + merged["description"].fillna("")
        + " "
        + merged["learning_objective"].fillna("")
        + " "
        + documents["question_type"].fillna("")
        + " "
        + documents["difficulty"].fillna("")
        + " "
        + documents["tags"].fillna("").str.replace("|", " ", regex=False)
        + " "
        + documents["prompt"].fillna("")
        + " "
        + documents["final_answer"].fillna("")
        + " "
        + documents["worked_solution"].fillna("")
    )
    return documents


class HybridSearchEngine:
    def __init__(
        self,
        *,
        documents: pd.DataFrame,
        bm25_retriever: BM25Retriever,
        vector_store: FaissVectorStore,
        query_engine: QueryUnderstandingEngine,
        ranking_weights: HybridRankingWeights | None = None,
    ) -> None:
        self.documents = documents.copy()
        self.documents_by_id = self.documents.set_index("doc_id")
        self.bm25_retriever = bm25_retriever
        self.vector_store = vector_store
        self.query_engine = query_engine
        self.ranking_weights = ranking_weights or HybridRankingWeights()

    @classmethod
    def from_artifacts(
        cls,
        *,
        data_dir: Path,
        index_dir: Path,
        embedding_backend: str | None = None,
        model_name: str | None = None,
    ) -> "HybridSearchEngine":
        concepts, concept_graph = load_concepts_and_graph(data_dir)

        vector_store = FaissVectorStore.load(
            input_dir=index_dir,
            embedding_backend=embedding_backend,
            model_name=model_name,
        )
        bm25_retriever = BM25Retriever(vector_store.documents)
        query_engine = QueryUnderstandingEngine(concepts=concepts, concept_graph=concept_graph)
        return cls(
            documents=vector_store.documents,
            bm25_retriever=bm25_retriever,
            vector_store=vector_store,
            query_engine=query_engine,
        )

    def search(self, query: str, top_k: int = 5) -> tuple[QueryIntent, list[SearchResult]]:
        intent = self.query_engine.understand(query)
        expanded_query = self._expanded_query(intent)

        bm25_hits = self.bm25_retriever.search(expanded_query, top_k=top_k * 3)
        vector_hits = self.vector_store.search(expanded_query, top_k=top_k * 3)

        bm25_scores = {hit.doc_id: hit.score for hit in bm25_hits}
        vector_scores = {hit.doc_id: hit.score for hit in vector_hits}
        candidate_doc_ids = list(dict.fromkeys([*bm25_scores.keys(), *vector_scores.keys()]))

        bm25_norm = self._normalize_scores(bm25_scores)
        vector_norm = self._normalize_scores(vector_scores)

        results: list[SearchResult] = []
        for doc_id in candidate_doc_ids:
            document = self.documents_by_id.loc[doc_id]
            exact_concept_match = document["concept_id"] in intent.detected_concept_ids
            concept_overlap = document["concept_id"] in intent.expanded_concept_ids
            difficulty_match = (
                intent.requested_difficulty is not None
                and document["difficulty"] == intent.requested_difficulty
            )
            w = self.ranking_weights
            hybrid_score = (
                w.bm25 * bm25_norm.get(doc_id, 0.0)
                + w.vector * vector_norm.get(doc_id, 0.0)
                + w.exact_concept * float(exact_concept_match)
                + w.concept_overlap * float(concept_overlap)
                + w.difficulty_match * float(difficulty_match)
            )

            results.append(
                SearchResult(
                    doc_id=doc_id,
                    question_id=str(document["question_id"]),
                    chapter_name=str(document["chapter_name"]),
                    concept_name=str(document["concept_name"]),
                    difficulty=str(document["difficulty"]),
                    prompt=str(document["prompt"]),
                    final_answer=str(document["final_answer"]),
                    bm25_score=float(bm25_scores.get(doc_id, 0.0)),
                    vector_score=float(vector_scores.get(doc_id, 0.0)),
                    hybrid_score=float(hybrid_score),
                    exact_concept_match=exact_concept_match,
                    concept_overlap=concept_overlap,
                    difficulty_match=difficulty_match,
                )
            )

        ranked = sorted(results, key=lambda item: item.hybrid_score, reverse=True)[:top_k]
        return intent, ranked

    def _expanded_query(self, intent: QueryIntent) -> str:
        segments = [intent.original_query, *intent.expanded_terms]
        return " ".join(segment for segment in segments if segment)

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        min_score = min(scores.values())
        max_score = max(scores.values())
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id in scores}
        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores.items()
        }


def build_hybrid_search_engine(
    *,
    documents: pd.DataFrame,
    concepts: pd.DataFrame,
    concept_graph,
    embedding_backend: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
) -> HybridSearchEngine:
    bm25_retriever = BM25Retriever(documents)
    embedder = build_embedder(
        embedding_backend=embedding_backend,
        model_name=model_name,
    )
    vector_store = FaissVectorStore.build(documents=documents, embedder=embedder)
    query_engine = QueryUnderstandingEngine(concepts=concepts, concept_graph=concept_graph)
    return HybridSearchEngine(
        documents=documents,
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        query_engine=query_engine,
    )
