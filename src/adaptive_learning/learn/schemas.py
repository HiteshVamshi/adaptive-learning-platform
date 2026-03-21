from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LearnResourceRecord:
    """Curated external learn asset (video, slides, article) tagged to curriculum."""

    resource_id: str
    subject: str
    concept_id: str
    chapter_id: str
    chapter_name: str
    concept_name: str
    resource_type: str
    title: str
    url: str
    source: str
    duration_min: int | None
    difficulty: str
    order_index: int
    license: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
