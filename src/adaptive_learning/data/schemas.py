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
    source_document: str = "bootstrap_seed"
    source_url: str = ""
    source_kind: str = "curated"
    syllabus_session: str = ""
    unit_name: str = ""
    unit_marks: int | None = None
    periods: int | None = None

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["tags"] = "|".join(self.tags)
        return record


@dataclass(frozen=True)
class SyllabusTopicRecord:
    topic_id: str
    topic_name: str
    class_level: str
    syllabus_session: str
    unit_id: str
    unit_name: str
    unit_marks: int
    chapter_id: str
    chapter_name: str
    periods: int
    official_text: str
    source_lines: str
    source_document: str
    source_url: str
    source_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TextbookSectionRecord:
    section_id: str
    class_level: str
    chapter_number: int
    chapter_id: str
    chapter_name: str
    section_number: str
    section_title: str
    start_page: int
    chapter_pdf_url: str
    source_document: str
    source_url: str
    source_lines: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TheoryContentRecord:
    content_id: str
    class_level: str
    chapter_id: str
    chapter_name: str
    concept_id: str
    concept_name: str
    title: str
    content_type: str
    summary_text: str
    official_syllabus_text: str
    textbook_sections: list[str]
    textbook_section_ids: list[str]
    source_document: str
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["textbook_sections"] = "|".join(self.textbook_sections)
        record["textbook_section_ids"] = "|".join(self.textbook_section_ids)
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
