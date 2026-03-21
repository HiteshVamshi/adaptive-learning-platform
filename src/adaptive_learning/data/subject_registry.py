"""Register subject dataset builders for extension without growing if-chains."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SubjectBuilder = Callable[[], Any]

_BUILDERS: dict[str, SubjectBuilder] = {}


def register_subject(subject_id: str, builder: SubjectBuilder) -> None:
    normalized = subject_id.strip().lower()
    _BUILDERS[normalized] = builder


def supported_subjects() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def generate_subject_dataset(subject: str) -> Any:
    normalized = subject.strip().lower()
    builder = _BUILDERS.get(normalized)
    if builder is None:
        raise ValueError(
            f"Unsupported subject '{subject}'. Expected one of: {', '.join(supported_subjects())}."
        )
    return builder()
