from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from adaptive_learning.search.text_utils import significant_tokens


@dataclass(frozen=True)
class VectorSearchResult:
    doc_id: str
    score: float
    rank: int


class BaseEmbedder:
    backend_name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(BaseEmbedder):
    def __init__(self, dimension: int = 256) -> None:
        self.backend_name = "hash"
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in significant_tokens(text):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % self.dimension
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                vectors[row_index, bucket] += sign
            norm = np.linalg.norm(vectors[row_index])
            if norm > 0:
                vectors[row_index] /= norm
        return vectors


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.backend_name = "sentence_transformer"
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


def build_embedder(
    *,
    embedding_backend: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
) -> BaseEmbedder:
    if embedding_backend == "hash":
        return HashEmbedder()
    if embedding_backend == "sentence_transformer":
        return SentenceTransformerEmbedder(model_name=model_name)

    try:
        return SentenceTransformerEmbedder(model_name=model_name)
    except Exception:
        return HashEmbedder()


class FaissVectorStore:
    def __init__(
        self,
        *,
        index: faiss.Index,
        documents: pd.DataFrame,
        embedder: BaseEmbedder,
    ) -> None:
        self.index = index
        self.documents = documents.reset_index(drop=True)
        self.embedder = embedder

    @classmethod
    def build(cls, *, documents: pd.DataFrame, embedder: BaseEmbedder) -> "FaissVectorStore":
        embeddings = embedder.encode(documents["search_text"].tolist())
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index=index, documents=documents, embedder=embedder)

    def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        query_vector = self.embedder.encode([query])
        top_k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_vector, top_k)

        results: list[VectorSearchResult] = []
        for rank, (doc_index, score) in enumerate(
            zip(indices[0], scores[0], strict=False),
            start=1,
        ):
            if doc_index < 0:
                continue
            doc_id = self.documents.iloc[int(doc_index)]["doc_id"]
            results.append(
                VectorSearchResult(doc_id=doc_id, score=float(score), rank=rank)
            )
        return results

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_dir / "faiss.index"))
        self.documents.to_csv(output_dir / "documents.csv", index=False)

        metadata = {
            "embedding_backend": self.embedder.backend_name,
            "embedding_dimension": self.embedder.dimension,
        }
        if hasattr(self.embedder, "model_name"):
            metadata["embedding_model_name"] = getattr(self.embedder, "model_name")

        with (output_dir / "vector_store_meta.json").open("w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2)

    @classmethod
    def load(
        cls,
        *,
        input_dir: Path,
        embedding_backend: str | None = None,
        model_name: str | None = None,
    ) -> "FaissVectorStore":
        documents = pd.read_csv(input_dir / "documents.csv")
        with (input_dir / "vector_store_meta.json").open("r", encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)

        resolved_backend = embedding_backend or metadata["embedding_backend"]
        resolved_model_name = model_name or metadata.get("embedding_model_name", "all-MiniLM-L6-v2")
        embedder = build_embedder(
            embedding_backend=resolved_backend,
            model_name=resolved_model_name,
        )
        index = faiss.read_index(str(input_dir / "faiss.index"))
        return cls(index=index, documents=documents, embedder=embedder)
