from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DifficultyExampleRecord:
    example_id: str
    source: str
    concept_id: str
    concept_name: str
    chapter_name: str
    difficulty: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptationComparisonRecord:
    concept_id: str
    concept_name: str
    target_difficulty: str
    baseline_prompt: str
    baseline_prediction: str
    baseline_target_probability: float
    adapted_prompt: str
    adapted_prediction: str
    adapted_target_probability: float
    adaptation_notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
