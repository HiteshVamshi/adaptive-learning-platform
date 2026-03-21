from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ContentGenerator(Protocol):
    """Contract for template- or model-backed generators used by the UI and agents."""

    backend_name: str

    def generate_question(self, *, concept_id: str, difficulty: str) -> tuple[Any, str]: ...

    def generate_explanation(self, *, concept_id: str, audience_level: str = "class_10_student") -> tuple[Any, str]: ...

    def generate_summary(self, *, scope_type: str, scope_id: str) -> tuple[Any, str]: ...
