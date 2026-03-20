from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ConceptRecord:
    concept_id: str
    name: str
    node_type: str
    chapter_id: str
    chapter_name: str
    parent_concept_id: str | None
    class_level: str
    description: str
    learning_objective: str
    tags: list[str]
    difficulty_band: str
    order_index: int

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["tags"] = "|".join(self.tags)
        return record


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    chapter_id: str
    chapter_name: str
    concept_id: str
    concept_name: str
    difficulty: str
    question_type: str
    prompt: str
    answer: str
    tags: list[str]
    estimated_time_sec: int
    source: str = "synthetic_curated"

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["tags"] = "|".join(self.tags)
        return record


@dataclass(frozen=True)
class SolutionRecord:
    question_id: str
    final_answer: str
    worked_solution: str
    explanation_style: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationshipRecord:
    source_concept_id: str
    target_concept_id: str
    relation_type: str
    weight: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
