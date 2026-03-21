from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecommendationWeights:
    weakness: float = 0.48
    difficulty_alignment: float = 0.22
    graph_priority: float = 0.18
    novelty: float = 0.12


@dataclass(frozen=True)
class RecommendationConfig:
    """Scoring mix and explicit cold-start defaults when no mastery row exists."""

    weights: RecommendationWeights = field(default_factory=RecommendationWeights)
    cold_start_mastery: float = 0.0
    cold_start_band: str = "needs_support"
