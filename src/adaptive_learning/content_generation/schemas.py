from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GeneratedQuestionRecord:
    concept_id: str
    concept_name: str
    chapter_name: str
    difficulty: str
    prompt: str
    final_answer: str
    explanation: str
    tags: list[str]
    backend: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = "|".join(self.tags)
        return payload


@dataclass(frozen=True)
class GeneratedExplanationRecord:
    concept_id: str
    concept_name: str
    chapter_name: str
    audience_level: str
    explanation: str
    key_points: list[str]
    backend: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key_points"] = "|".join(self.key_points)
        return payload


@dataclass(frozen=True)
class GeneratedSummaryRecord:
    scope_type: str
    scope_id: str
    title: str
    summary: str
    bullet_points: list[str]
    backend: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bullet_points"] = "|".join(self.bullet_points)
        return payload
