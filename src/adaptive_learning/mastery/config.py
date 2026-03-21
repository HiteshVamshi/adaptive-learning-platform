from __future__ import annotations

from dataclasses import dataclass, field


def _default_difficulty_weight() -> dict[str, float]:
    return {"easy": 1.0, "medium": 1.15, "hard": 1.3}


def _default_challenge_score() -> dict[str, float]:
    return {"easy": 0.55, "medium": 0.75, "hard": 1.0}


@dataclass(frozen=True)
class MasteryModelConfig:
    """Tunable mastery blend and difficulty weighting (direct signal vs graph support)."""

    direct_blend_direct: float = 0.92
    direct_blend_graph: float = 0.08
    difficulty_weight: dict[str, float] = field(default_factory=_default_difficulty_weight)
    challenge_score: dict[str, float] = field(default_factory=_default_challenge_score)
