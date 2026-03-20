from __future__ import annotations

import networkx as nx
import pandas as pd

from adaptive_learning.recommendation.schemas import RecommendationRecord


TARGET_DIFFICULTY_ORDER = {
    "needs_support": ["easy", "medium", "hard"],
    "developing": ["medium", "hard", "easy"],
    "mastered": ["hard", "medium", "easy"],
}

DIFFICULTY_ALIGNMENT = {
    "easy": {"easy": 1.0, "medium": 0.6, "hard": 0.25},
    "medium": {"easy": 0.7, "medium": 1.0, "hard": 0.65},
    "hard": {"easy": 0.3, "medium": 0.75, "hard": 1.0},
}


class RecommendationEngine:
    def __init__(
        self,
        *,
        questions: pd.DataFrame,
        mastery_snapshot: pd.DataFrame,
        attempts: pd.DataFrame,
        concept_graph: nx.MultiDiGraph,
    ) -> None:
        self.questions = questions.copy()
        self.mastery_snapshot = mastery_snapshot.copy()
        self.attempts = attempts.copy()
        self.concept_graph = concept_graph
        self.mastery_by_concept = self.mastery_snapshot.set_index("concept_id").to_dict(orient="index")
        self.question_attempt_counts = self.attempts["question_id"].value_counts().to_dict()

    def recommend(self, *, top_k: int = 10) -> pd.DataFrame:
        scored_rows: list[RecommendationRecord] = []

        for question in self.questions.to_dict(orient="records"):
            concept_id = str(question["concept_id"])
            mastery = self.mastery_by_concept.get(concept_id, {})
            graph_adjusted_mastery = float(mastery.get("graph_adjusted_mastery", 0.0))
            mastery_band = str(mastery.get("mastery_band", "needs_support"))

            weakness_score = 1.0 - graph_adjusted_mastery
            target_difficulty = TARGET_DIFFICULTY_ORDER[mastery_band][0]
            difficulty_alignment_score = DIFFICULTY_ALIGNMENT[target_difficulty][str(question["difficulty"])]
            graph_priority_score = self._graph_priority(concept_id)
            novelty_score = self._novelty_score(str(question["question_id"]))

            recommendation_score = (
                0.48 * weakness_score
                + 0.22 * difficulty_alignment_score
                + 0.18 * graph_priority_score
                + 0.12 * novelty_score
            )

            scored_rows.append(
                RecommendationRecord(
                    user_id=str(mastery.get("user_id", "")),
                    recommendation_rank=0,
                    question_id=str(question["question_id"]),
                    chapter_name=str(question["chapter_name"]),
                    concept_id=concept_id,
                    concept_name=str(question["concept_name"]),
                    difficulty=str(question["difficulty"]),
                    recommendation_score=round(float(recommendation_score), 4),
                    weakness_score=round(float(weakness_score), 4),
                    difficulty_alignment_score=round(float(difficulty_alignment_score), 4),
                    graph_priority_score=round(float(graph_priority_score), 4),
                    novelty_score=round(float(novelty_score), 4),
                    rationale=self._build_rationale(
                        concept_name=str(question["concept_name"]),
                        mastery_band=mastery_band,
                        graph_adjusted_mastery=graph_adjusted_mastery,
                        question_difficulty=str(question["difficulty"]),
                        target_difficulty=target_difficulty,
                        novelty_score=novelty_score,
                        graph_priority_score=graph_priority_score,
                    ),
                )
            )

        recommendations = pd.DataFrame([row.to_dict() for row in scored_rows])
        recommendations = recommendations.sort_values(
            by=[
                "recommendation_score",
                "weakness_score",
                "difficulty_alignment_score",
                "novelty_score",
            ],
            ascending=[False, False, False, False],
            ignore_index=True,
        )

        recommendations = self._limit_duplicate_concepts(recommendations, top_k=top_k)
        recommendations["recommendation_rank"] = range(1, len(recommendations) + 1)
        return recommendations

    def _limit_duplicate_concepts(self, recommendations: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
        selected_rows = []
        concept_counts: dict[str, int] = {}

        for row in recommendations.to_dict(orient="records"):
            concept_id = str(row["concept_id"])
            current_count = concept_counts.get(concept_id, 0)
            if current_count >= 2:
                continue
            selected_rows.append(row)
            concept_counts[concept_id] = current_count + 1
            if len(selected_rows) == top_k:
                break

        return pd.DataFrame(selected_rows)

    def _graph_priority(self, concept_id: str) -> float:
        prerequisite_needs = []
        downstream_needs = []

        for source_id, _, edge_data in self.concept_graph.in_edges(concept_id, data=True):
            if edge_data.get("relation_type") == "prerequisite":
                prerequisite_needs.append(self._concept_need(source_id))

        for _, target_id, edge_data in self.concept_graph.out_edges(concept_id, data=True):
            relation_type = edge_data.get("relation_type")
            if relation_type == "prerequisite":
                downstream_needs.append(self._concept_need(target_id))
            elif relation_type == "related":
                downstream_needs.append(0.6 * self._concept_need(target_id))

        prerequisite_boost = sum(prerequisite_needs) / len(prerequisite_needs) if prerequisite_needs else 0.0
        downstream_boost = sum(downstream_needs) / len(downstream_needs) if downstream_needs else 0.0
        return min(1.0, 0.65 * prerequisite_boost + 0.35 * downstream_boost)

    def _concept_need(self, concept_id: str) -> float:
        mastery = self.mastery_by_concept.get(concept_id)
        if mastery is None:
            return 0.5
        return 1.0 - float(mastery.get("graph_adjusted_mastery", 0.0))

    def _novelty_score(self, question_id: str) -> float:
        seen_count = int(self.question_attempt_counts.get(question_id, 0))
        if seen_count == 0:
            return 1.0
        if seen_count == 1:
            return 0.65
        if seen_count == 2:
            return 0.4
        return 0.15

    @staticmethod
    def _build_rationale(
        *,
        concept_name: str,
        mastery_band: str,
        graph_adjusted_mastery: float,
        question_difficulty: str,
        target_difficulty: str,
        novelty_score: float,
        graph_priority_score: float,
    ) -> str:
        novelty_text = "new question" if novelty_score >= 0.95 else "lightly repeated practice"
        graph_text = "high graph priority" if graph_priority_score >= 0.55 else "local concept practice"
        return (
            f"{concept_name} is currently {mastery_band} "
            f"(mastery={graph_adjusted_mastery:.2f}); "
            f"recommended difficulty is {target_difficulty} and this question is {question_difficulty}; "
            f"{novelty_text}; {graph_text}."
        )
