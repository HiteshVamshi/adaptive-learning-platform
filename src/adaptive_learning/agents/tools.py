from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adaptive_learning.learn.io import load_learn_resources
from adaptive_learning.learn.selector import format_learn_summary, select_learn_plan
from networkx.readwrite import json_graph

from adaptive_learning.config import rag_generator_backend
from adaptive_learning.content_generation.generator import build_content_generator
from adaptive_learning.content_generation.pipeline import load_generation_context
from adaptive_learning.rag.pipeline import RAGEngine
from adaptive_learning.search.hybrid_search import HybridSearchEngine


@dataclass
class AgentToolbox:
    data_dir: Path
    search_index_dir: Path
    rag_index_dir: Path
    mastery_dir: Path
    recommendation_dir: Path

    def __post_init__(self) -> None:
        self._search_engine = None
        self._rag_engine = None
        self._concepts = None
        self._questions = None
        self._concept_graph = None
        self._mastery_snapshot = None
        self._mastery_history = None
        self._recommendations = None
        self._content_generator = None

    @property
    def search_engine(self) -> HybridSearchEngine:
        if self._search_engine is None:
            self._search_engine = HybridSearchEngine.from_artifacts(
                data_dir=self.data_dir,
                index_dir=self.search_index_dir,
            )
        return self._search_engine

    @property
    def rag_engine(self) -> RAGEngine:
        if self._rag_engine is None:
            self._rag_engine = RAGEngine.from_artifacts(
                data_dir=self.data_dir,
                index_dir=self.rag_index_dir,
                generator_backend=rag_generator_backend(),
            )
        return self._rag_engine

    @property
    def concepts(self) -> pd.DataFrame:
        if self._concepts is None:
            self._concepts = pd.read_csv(self.data_dir / "concepts.csv")
        return self._concepts

    @property
    def questions(self) -> pd.DataFrame:
        if self._questions is None:
            self._questions = pd.read_csv(self.data_dir / "questions.csv")
        return self._questions

    @property
    def concept_graph(self):
        if self._concept_graph is None:
            with (self.data_dir / "concept_graph.json").open("r", encoding="utf-8") as file_obj:
                self._concept_graph = json_graph.node_link_graph(json.load(file_obj), multigraph=True)
        return self._concept_graph

    @property
    def mastery_snapshot(self) -> pd.DataFrame:
        if self._mastery_snapshot is None:
            self._mastery_snapshot = pd.read_csv(self.mastery_dir / "mastery_snapshot.csv")
        return self._mastery_snapshot

    @property
    def mastery_history(self) -> pd.DataFrame:
        if self._mastery_history is None:
            self._mastery_history = pd.read_csv(self.mastery_dir / "mastery_history.csv")
        return self._mastery_history

    @property
    def recommendations(self) -> pd.DataFrame:
        if self._recommendations is None:
            self._recommendations = pd.read_csv(self.recommendation_dir / "recommendations.csv")
        return self._recommendations

    @property
    def content_generator(self):
        if self._content_generator is None:
            context = load_generation_context(data_dir=self.data_dir)
            self._content_generator = build_content_generator(context=context, backend="grounded")
        return self._content_generator

    def hybrid_search(self, query: str, top_k: int = 3) -> dict:
        try:
            intent, results = self.search_engine.search(query, top_k=top_k)
            return {
                "intent": {
                    "requested_difficulty": intent.requested_difficulty,
                    "detected_concepts": intent.detected_concept_ids,
                    "expanded_concepts": intent.expanded_concept_ids,
                    "expanded_terms": intent.expanded_terms,
                },
                "results": [
                    {
                        "question_id": result.question_id,
                        "concept_name": result.concept_name,
                        "concept_id": self._concept_id_for_question(result.question_id),
                        "chapter_name": result.chapter_name,
                        "difficulty": result.difficulty,
                        "prompt": result.prompt,
                        "final_answer": result.final_answer,
                        "hybrid_score": round(result.hybrid_score, 4),
                    }
                    for result in results
                ],
            }
        except Exception as exc:
            return _tool_error_payload(
                "hybrid_search",
                str(exc),
                {
                    "intent": {
                        "requested_difficulty": None,
                        "detected_concepts": [],
                        "expanded_concepts": [],
                        "expanded_terms": [],
                    },
                    "results": [],
                },
            )

    def rag_answer(self, query: str, top_k: int = 4) -> dict:
        try:
            response = self.rag_engine.answer_question(query, top_k=top_k)
            return {
                "answer": response.answer.answer,
                "retrieved_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_type": chunk.chunk_type,
                        "concept_name": chunk.concept_name,
                        "chapter_name": chunk.chapter_name,
                        "score": round(chunk.score, 4),
                    }
                    for chunk in response.retrieved_chunks
                ],
            }
        except Exception as exc:
            return _tool_error_payload(
                "rag_answer",
                str(exc),
                {"answer": "", "retrieved_chunks": []},
            )

    def concept_mastery(self, concept_id: str) -> dict:
        row = self.mastery_snapshot[self.mastery_snapshot["concept_id"] == concept_id]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def weak_concepts(self, top_k: int = 5) -> list[dict]:
        rows = self.mastery_snapshot.sort_values(by="graph_adjusted_mastery", ascending=True).head(top_k)
        return rows[
            ["concept_id", "concept_name", "chapter_name", "graph_adjusted_mastery", "mastery_band", "explanation"]
        ].to_dict(orient="records")

    def recommendation_list(self, top_k: int = 5, concept_id: str | None = None) -> list[dict]:
        rows = self.recommendations.copy()
        if concept_id:
            rows = rows[rows["concept_id"] == concept_id]
        return rows.head(top_k).to_dict(orient="records")

    def concept_neighbors(self, concept_id: str) -> dict:
        if concept_id not in self.concept_graph:
            return {"concept_id": concept_id, "neighbors": []}
        neighbors = []
        for source_id, _, edge_data in self.concept_graph.in_edges(concept_id, data=True):
            if edge_data.get("relation_type") in {"prerequisite", "related"}:
                neighbors.append(
                    {
                        "concept_id": source_id,
                        "concept_name": self.concept_graph.nodes[source_id].get("name"),
                        "relation_type": edge_data.get("relation_type"),
                    }
                )
        for _, target_id, edge_data in self.concept_graph.out_edges(concept_id, data=True):
            if edge_data.get("relation_type") in {"prerequisite", "related"}:
                neighbors.append(
                    {
                        "concept_id": target_id,
                        "concept_name": self.concept_graph.nodes[target_id].get("name"),
                        "relation_type": edge_data.get("relation_type"),
                    }
                )
        return {"concept_id": concept_id, "neighbors": neighbors}

    def generate_question(self, concept_id: str, difficulty: str) -> dict:
        record, prompt = self.content_generator.generate_question(concept_id=concept_id, difficulty=difficulty)
        return {"prompt_template": prompt, "output": record.to_dict()}

    def generate_explanation(self, concept_id: str) -> dict:
        record, prompt = self.content_generator.generate_explanation(concept_id=concept_id)
        return {"prompt_template": prompt, "output": record.to_dict()}

    def rewrite_query(self, query: str) -> dict:
        search_payload = self.hybrid_search(query, top_k=3)
        if search_payload.get("_tool_error"):
            return search_payload
        intent = search_payload["intent"]
        concept_names = []
        for concept_id in intent["detected_concepts"]:
            row = self.concepts[self.concepts["concept_id"] == concept_id]
            if not row.empty:
                concept_names.append(str(row.iloc[0]["name"]))
        rewritten = query
        if concept_names:
            rewritten = f"{query} concept:{', '.join(concept_names)}"
        if intent["requested_difficulty"]:
            rewritten = f"{rewritten} difficulty:{intent['requested_difficulty']}"
        return {
            "rewritten_query": rewritten.strip(),
            "intent": intent,
            "top_results": search_payload["results"],
        }

    def _concept_id_for_question(self, question_id: str) -> str:
        row = self.questions[self.questions["question_id"] == question_id]
        if row.empty:
            return ""
        return str(row.iloc[0]["concept_id"])

    def learn_recommendations(
        self,
        *,
        live_mastery_snapshot: pd.DataFrame | None = None,
        focus_concept_ids: list[str] | None = None,
        max_concepts: int = 6,
    ) -> dict:
        artifacts_dir = self.data_dir.parent
        resources_df = load_learn_resources(artifacts_dir)
        if resources_df.empty:
            return {
                "summary": "No learn_resources.csv yet. Run: python scripts/build_learn_index.py --subject <math|science> or build_all / generate_data.",
                "items": [],
            }

        snapshot = live_mastery_snapshot if live_mastery_snapshot is not None else self.mastery_snapshot
        if focus_concept_ids:
            ids = {str(x) for x in focus_concept_ids}
            snapshot = snapshot[snapshot["concept_id"].astype(str).isin(ids)].copy()
            if snapshot.empty:
                weak = self.weak_concepts(top_k=max_concepts)
                if weak:
                    wids = [str(w["concept_id"]) for w in weak]
                    snapshot = self.mastery_snapshot[
                        self.mastery_snapshot["concept_id"].astype(str).isin(wids)
                    ].copy()

        plan = select_learn_plan(
            live_snapshot=snapshot,
            resources_df=resources_df,
            concepts=self.concepts,
            max_concepts=max_concepts,
        )
        items: list[dict] = []
        for concept_row, res_df in plan:
            for _, r in res_df.iterrows():
                items.append(
                    {
                        "concept_id": concept_row.get("concept_id"),
                        "concept_name": concept_row.get("concept_name"),
                        "mastery_band": concept_row.get("mastery_band"),
                        "resource_id": r["resource_id"],
                        "resource_type": r["resource_type"],
                        "title": r["title"],
                        "url": r["url"],
                        "source": r["source"],
                    }
                )
        return {
            "summary": format_learn_summary(plan),
            "items": items,
        }


def _tool_error_payload(tool: str, message: str, payload: dict) -> dict:
    merged = dict(payload)
    merged["_tool_error"] = True
    merged["tool"] = tool
    merged["message"] = message
    return merged
