from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rank_bm25 import BM25Okapi

from adaptive_learning.search.text_utils import significant_tokens


@dataclass(frozen=True)
class RetrievalScore:
    doc_id: str
    score: float
    rank: int


class BM25Retriever:
    def __init__(self, documents: pd.DataFrame) -> None:
        self.documents = documents.reset_index(drop=True)
        self.doc_ids = self.documents["doc_id"].tolist()
        self.corpus_tokens = [
            significant_tokens(text) for text in self.documents["search_text"].tolist()
        ]
        self.index = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalScore]:
        query_tokens = significant_tokens(query)
        if not query_tokens:
            query_tokens = ["mathematics"]

        scores = self.index.get_scores(query_tokens)
        ranked_pairs = sorted(
            zip(self.doc_ids, scores, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return [
            RetrievalScore(doc_id=doc_id, score=float(score), rank=rank)
            for rank, (doc_id, score) in enumerate(ranked_pairs, start=1)
        ]
