from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    user_id: str
    simulation_step: int
    timestamp: str
    question_id: str
    concept_id: str
    concept_name: str
    chapter_name: str
    difficulty: str
    correct: int
    response_time_sec: int
    expected_time_sec: int
    source: str = "simulated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MasterySnapshotRecord:
    user_id: str
    concept_id: str
    concept_name: str
    chapter_name: str
    attempts_count: int
    correct_count: int
    weighted_accuracy: float
    completion: float
    speed: float
    challenge: float
    confidence: float
    direct_mastery: float
    graph_support: float
    graph_adjusted_mastery: float
    mastery_band: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MasteryHistoryRecord:
    user_id: str
    simulation_step: int
    timestamp: str
    concept_id: str
    concept_name: str
    chapter_name: str
    attempts_count: int
    direct_mastery: float
    graph_adjusted_mastery: float
    mastery_band: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
