"""Lightweight checks before writing curriculum artifacts."""

from __future__ import annotations

from adaptive_learning.data.generator import GeneratedDataset
from adaptive_learning.kg.concept_graph import validate_concept_graph

_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
_NODE_TYPES = frozenset({"concept", "chapter", "unit"})


def validate_generated_dataset(dataset: GeneratedDataset) -> None:
    """Raise ValueError if required columns or referential integrity fail."""
    _require_columns(dataset.concepts, ["concept_id", "node_type", "chapter_id", "name"], "concepts")
    _require_columns(
        dataset.questions,
        ["question_id", "concept_id", "chapter_id", "difficulty", "prompt"],
        "questions",
    )
    _require_columns(dataset.solutions, ["question_id", "final_answer"], "solutions")
    _require_columns(
        dataset.relationships,
        ["source_concept_id", "target_concept_id", "relation_type"],
        "relationships",
    )

    bad_nodes = set(dataset.concepts["node_type"].unique()) - _NODE_TYPES
    if bad_nodes:
        raise ValueError(f"concepts.node_type has unknown values: {sorted(bad_nodes)}")

    bad_diff = set(dataset.questions["difficulty"].unique()) - _DIFFICULTIES
    if bad_diff:
        raise ValueError(f"questions.difficulty has unknown values: {sorted(bad_diff)}")

    q_ids = set(dataset.questions["question_id"].astype(str))
    sol_q = set(dataset.solutions["question_id"].astype(str))
    missing_sol = q_ids - sol_q
    if missing_sol:
        raise ValueError(f"solutions missing for question_id(s): {sorted(missing_sol)[:5]}...")

    concept_ids = set(dataset.concepts["concept_id"].astype(str))
    orphan_q = set(dataset.questions["concept_id"].astype(str)) - concept_ids
    if orphan_q:
        raise ValueError(f"questions reference unknown concept_id(s): {sorted(orphan_q)[:8]}")

    graph_issues = validate_concept_graph(
        dataset.concept_graph,
        concepts=dataset.concepts,
        relationships=dataset.relationships,
    )
    if graph_issues:
        raise ValueError("concept graph validation failed:\n- " + "\n- ".join(graph_issues[:20]))


def _require_columns(frame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
