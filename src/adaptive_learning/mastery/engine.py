from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from adaptive_learning.mastery.schemas import MasteryHistoryRecord


DIFFICULTY_WEIGHT = {
    "easy": 1.0,
    "medium": 1.15,
    "hard": 1.3,
}

CHALLENGE_SCORE = {
    "easy": 0.55,
    "medium": 0.75,
    "hard": 1.0,
}


@dataclass(frozen=True)
class MasteryRunResult:
    snapshot: pd.DataFrame
    history: pd.DataFrame


class ConceptMasteryEngine:
    def __init__(self, *, concepts: pd.DataFrame, concept_graph: nx.MultiDiGraph) -> None:
        self.concepts = concepts[concepts["node_type"] == "concept"].copy()
        self.concepts_by_id = self.concepts.set_index("concept_id")
        self.concept_graph = concept_graph

    def run(self, *, attempts: pd.DataFrame) -> MasteryRunResult:
        attempts = attempts.sort_values(by=["simulation_step", "timestamp"], ignore_index=True)

        history_rows: list[MasteryHistoryRecord] = []
        latest_snapshot = pd.DataFrame()

        for _, attempt in attempts.iterrows():
            step_attempts = attempts[attempts["simulation_step"] <= attempt["simulation_step"]]
            latest_snapshot = self.compute_snapshot(step_attempts)

            for row in latest_snapshot.to_dict(orient="records"):
                history_rows.append(
                    MasteryHistoryRecord(
                        user_id=str(row["user_id"]),
                        simulation_step=int(attempt["simulation_step"]),
                        timestamp=str(attempt["timestamp"]),
                        concept_id=str(row["concept_id"]),
                        concept_name=str(row["concept_name"]),
                        chapter_name=str(row["chapter_name"]),
                        attempts_count=int(row["attempts_count"]),
                        direct_mastery=float(row["direct_mastery"]),
                        graph_adjusted_mastery=float(row["graph_adjusted_mastery"]),
                        mastery_band=str(row["mastery_band"]),
                    )
                )

        history = pd.DataFrame([record.to_dict() for record in history_rows])
        return MasteryRunResult(snapshot=latest_snapshot, history=history)

    def compute_snapshot(self, attempts: pd.DataFrame) -> pd.DataFrame:
        concept_records: list[dict] = []

        for concept in self.concepts.to_dict(orient="records"):
            concept_id = str(concept["concept_id"])
            concept_attempts = attempts[attempts["concept_id"] == concept_id].copy()
            concept_records.append(self._compute_direct_mastery(concept, concept_attempts))

        snapshot = pd.DataFrame(concept_records)
        if snapshot.empty:
            return snapshot

        snapshot["graph_support"] = snapshot["concept_id"].map(
            lambda concept_id: self._graph_support(concept_id, snapshot)
        )
        snapshot["graph_adjusted_mastery"] = (
            0.92 * snapshot["direct_mastery"] + 0.08 * snapshot["graph_support"]
        ).clip(0.0, 1.0)
        snapshot["mastery_band"] = snapshot["graph_adjusted_mastery"].map(self._band_for_score)
        snapshot["explanation"] = snapshot.apply(self._build_explanation, axis=1)
        return snapshot.sort_values(
            by=["graph_adjusted_mastery", "direct_mastery"],
            ascending=[False, False],
            ignore_index=True,
        )

    def _compute_direct_mastery(self, concept: dict, concept_attempts: pd.DataFrame) -> dict:
        if concept_attempts.empty:
            return {
                "user_id": "",
                "concept_id": concept["concept_id"],
                "concept_name": concept["name"],
                "chapter_name": concept["chapter_name"],
                "attempts_count": 0,
                "correct_count": 0,
                "weighted_accuracy": 0.0,
                "completion": 0.0,
                "speed": 0.0,
                "challenge": 0.0,
                "confidence": 0.0,
                "direct_mastery": 0.0,
            }

        weighted_correct = 0.0
        weighted_total = 0.0
        challenge_scores = []
        speed_scores = []

        for attempt in concept_attempts.to_dict(orient="records"):
            difficulty = str(attempt["difficulty"])
            weight = DIFFICULTY_WEIGHT[difficulty]
            weighted_total += weight
            weighted_correct += weight * int(attempt["correct"])
            challenge_scores.append(CHALLENGE_SCORE[difficulty] if int(attempt["correct"]) else 0.0)

            expected_time = max(1, int(attempt["expected_time_sec"]))
            response_time = max(1, int(attempt["response_time_sec"]))
            speed_ratio = np.clip(expected_time / response_time, 0.45, 1.25)
            speed_scores.append(float((speed_ratio - 0.45) / (1.25 - 0.45)))

        weighted_accuracy = weighted_correct / weighted_total if weighted_total else 0.0
        attempts_count = int(len(concept_attempts))
        correct_count = int(concept_attempts["correct"].sum())
        completion = min(attempts_count / 5.0, 1.0)
        speed = float(np.mean(speed_scores)) if speed_scores else 0.0
        challenge = float(np.mean(challenge_scores)) if challenge_scores else 0.0
        confidence = min(attempts_count / 6.0, 1.0)

        recent_correctness = concept_attempts["correct"].tail(5).to_numpy(dtype=float)
        stability = 1.0 - float(np.std(recent_correctness)) if len(recent_correctness) > 1 else 1.0
        stability = float(np.clip(stability, 0.0, 1.0))

        raw_score = (
            0.5 * weighted_accuracy
            + 0.18 * completion
            + 0.12 * speed
            + 0.12 * challenge
            + 0.08 * stability
        )
        direct_mastery = raw_score * (0.6 + 0.4 * confidence)

        return {
            "user_id": str(concept_attempts.iloc[-1]["user_id"]),
            "concept_id": concept["concept_id"],
            "concept_name": concept["name"],
            "chapter_name": concept["chapter_name"],
            "attempts_count": attempts_count,
            "correct_count": correct_count,
            "weighted_accuracy": round(weighted_accuracy, 4),
            "completion": round(completion, 4),
            "speed": round(speed, 4),
            "challenge": round(challenge, 4),
            "confidence": round(confidence, 4),
            "direct_mastery": round(float(np.clip(direct_mastery, 0.0, 1.0)), 4),
        }

    def _graph_support(self, concept_id: str, snapshot: pd.DataFrame) -> float:
        score_by_concept = snapshot.set_index("concept_id")["direct_mastery"].to_dict()
        prerequisite_scores = []
        related_scores = []

        for source_id, _, edge_data in self.concept_graph.in_edges(concept_id, data=True):
            relation_type = edge_data.get("relation_type")
            if relation_type == "prerequisite" and source_id in score_by_concept:
                prerequisite_scores.append(score_by_concept[source_id])
            elif relation_type == "related" and source_id in score_by_concept:
                related_scores.append(score_by_concept[source_id])

        for _, target_id, edge_data in self.concept_graph.out_edges(concept_id, data=True):
            if edge_data.get("relation_type") == "related" and target_id in score_by_concept:
                related_scores.append(score_by_concept[target_id])

        prerequisite_support = float(np.mean(prerequisite_scores)) if prerequisite_scores else 0.0
        related_support = float(np.mean(related_scores)) if related_scores else 0.0

        if prerequisite_scores and related_scores:
            return round(0.75 * prerequisite_support + 0.25 * related_support, 4)
        if prerequisite_scores:
            return round(prerequisite_support, 4)
        if related_scores:
            return round(related_support, 4)
        return 0.0

    @staticmethod
    def _band_for_score(score: float) -> str:
        if score >= 0.75:
            return "mastered"
        if score >= 0.5:
            return "developing"
        return "needs_support"

    @staticmethod
    def _build_explanation(row: pd.Series) -> str:
        return (
            f"Attempts={int(row['attempts_count'])}, weighted_accuracy={row['weighted_accuracy']:.2f}, "
            f"completion={row['completion']:.2f}, speed={row['speed']:.2f}, "
            f"challenge={row['challenge']:.2f}, confidence={row['confidence']:.2f}, "
            f"graph_support={row['graph_support']:.2f}."
        )
