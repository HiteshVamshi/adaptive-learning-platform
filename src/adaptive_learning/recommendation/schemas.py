from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RecommendationRecord:
    user_id: str
    recommendation_rank: int
    question_id: str
    chapter_name: str
    concept_id: str
    concept_name: str
    difficulty: str
    recommendation_score: float
    weakness_score: float
    difficulty_alignment_score: float
    graph_priority_score: float
    novelty_score: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
