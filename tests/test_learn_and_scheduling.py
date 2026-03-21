from __future__ import annotations

import pandas as pd

from adaptive_learning.data.generator import generate_cbse_math_dataset
from adaptive_learning.learn.build_index import write_learn_index_for_subject
from adaptive_learning.learn.io import load_learn_resources
from adaptive_learning.learn.selector import select_learn_plan, select_resources_for_concept
from adaptive_learning.learn.sources.math_programmatic import math_learn_resource_seed
from adaptive_learning.learn.sources.science_programmatic import science_learn_resource_seed
from adaptive_learning.scheduling.spaced_repetition import build_spaced_repetition_queue
from pathlib import Path


def test_math_learn_seed_unique_ids() -> None:
    rows = math_learn_resource_seed()
    ids = [r.resource_id for r in rows]
    assert len(ids) == len(set(ids))


def test_science_learn_seed_unique_ids() -> None:
    rows = science_learn_resource_seed()
    ids = [r.resource_id for r in rows]
    assert len(ids) == len(set(ids))


def test_select_resources_respects_mastery_band() -> None:
    dataset = generate_cbse_math_dataset()
    rows = [r.to_dict() for r in math_learn_resource_seed()]
    resources_df = pd.DataFrame(rows)
    cid = "c_polynomial_degree_terms"
    chid = "ch_polynomials"
    snap_needs = pd.DataFrame(
        [
            {
                "concept_id": cid,
                "graph_adjusted_mastery": 0.2,
                "mastery_band": "needs_support",
                "attempts_count": 0,
            }
        ]
    )
    out_need = select_resources_for_concept(
        concept_id=cid,
        chapter_id=chid,
        live_snapshot=snap_needs,
        resources_df=resources_df,
    )
    assert len(out_need) >= 2

    snap_mastered = pd.DataFrame(
        [
            {
                "concept_id": cid,
                "graph_adjusted_mastery": 0.9,
                "mastery_band": "mastered",
                "attempts_count": 10,
            }
        ]
    )
    out_master = select_resources_for_concept(
        concept_id=cid,
        chapter_id=chid,
        live_snapshot=snap_mastered,
        resources_df=resources_df,
    )
    assert len(out_master) <= len(out_need)


def test_select_learn_plan_uses_chapter_merge(tmp_path: Path) -> None:
    dataset = generate_cbse_math_dataset()
    concepts = dataset.concepts[dataset.concepts["node_type"] == "concept"]
    write_learn_index_for_subject(artifacts_dir=tmp_path, subject="math")
    resources_df = load_learn_resources(tmp_path)
    snap = pd.DataFrame(
        [
            {
                "concept_id": str(concepts.iloc[0]["concept_id"]),
                "concept_name": str(concepts.iloc[0]["name"]),
                "chapter_name": str(concepts.iloc[0]["chapter_name"]),
                "graph_adjusted_mastery": 0.1,
                "mastery_band": "needs_support",
                "attempts_count": 0,
            }
        ]
    )
    plan = select_learn_plan(
        live_snapshot=snap,
        resources_df=resources_df,
        concepts=dataset.concepts,
        max_concepts=5,
    )
    assert isinstance(plan, list)


def test_spaced_repetition_orders_by_priority() -> None:
    live = pd.DataFrame(
        [
            {
                "concept_id": "c_a",
                "concept_name": "A",
                "chapter_name": "Ch1",
                "mastery_band": "mastered",
                "graph_adjusted_mastery": 0.95,
            },
            {
                "concept_id": "c_b",
                "concept_name": "B",
                "chapter_name": "Ch1",
                "mastery_band": "needs_support",
                "graph_adjusted_mastery": 0.2,
            },
        ]
    )
    attempts = pd.DataFrame(
        [
            {
                "concept_id": "c_a",
                "timestamp": "2020-01-01T00:00:00",
            },
            {
                "concept_id": "c_b",
                "timestamp": "2025-06-01T00:00:00",
            },
        ]
    )
    q = build_spaced_repetition_queue(
        user_attempts=attempts,
        live_snapshot=live,
        now=pd.Timestamp("2026-06-15T12:00:00Z"),
    )
    assert not q.empty
    assert q.iloc[0]["concept_id"] == "c_b"
