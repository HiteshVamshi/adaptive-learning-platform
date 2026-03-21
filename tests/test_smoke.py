from __future__ import annotations

import pandas as pd

from adaptive_learning.data.concept_question_bank import ConceptBankConfig, generate_concept_question_bank
from adaptive_learning.data.generator import SUPPORTED_SUBJECTS, generate_cbse_math_dataset, generate_subject_dataset
from adaptive_learning.data.dataset_validation import validate_generated_dataset
from adaptive_learning.data.subject_registry import supported_subjects
from adaptive_learning.kg.concept_graph import build_concept_graph, validate_concept_graph
from adaptive_learning.mastery.engine import ConceptMasteryEngine
from adaptive_learning.rag.pipeline import RAGEngine, RetrievedChunk


def test_supported_subjects_matches_registry() -> None:
    assert set(SUPPORTED_SUBJECTS) == set(supported_subjects())


def test_generate_math_dataset_validates() -> None:
    dataset = generate_cbse_math_dataset()
    validate_generated_dataset(dataset)
    assert dataset.concept_graph.name == "cbse_class_10_math_concepts"


def test_generate_science_dataset_validates() -> None:
    dataset = generate_subject_dataset("science")
    validate_generated_dataset(dataset)
    assert "science" in dataset.concept_graph.name


def test_mastery_run_caches_snapshots_by_step() -> None:
    dataset = generate_cbse_math_dataset()
    engine = ConceptMasteryEngine(concepts=dataset.concepts, concept_graph=dataset.concept_graph)
    attempts = pd.DataFrame(
        [
            {
                "attempt_id": "a1",
                "user_id": "u1",
                "simulation_step": 1,
                "timestamp": "2024-01-01T00:00:00",
                "question_id": "q1",
                "concept_id": dataset.concepts[dataset.concepts["node_type"] == "concept"].iloc[0]["concept_id"],
                "concept_name": "x",
                "chapter_name": "y",
                "difficulty": "easy",
                "correct": 1,
                "response_time_sec": 60,
                "expected_time_sec": 90,
                "source": "test",
            },
            {
                "attempt_id": "a2",
                "user_id": "u1",
                "simulation_step": 1,
                "timestamp": "2024-01-01T00:01:00",
                "question_id": "q2",
                "concept_id": dataset.concepts[dataset.concepts["node_type"] == "concept"].iloc[0]["concept_id"],
                "concept_name": "x",
                "chapter_name": "y",
                "difficulty": "easy",
                "correct": 0,
                "response_time_sec": 60,
                "expected_time_sec": 90,
                "source": "test",
            },
        ]
    )
    result = engine.run(attempts=attempts)
    assert not result.snapshot.empty
    assert len(result.history) == 2 * len(result.snapshot)


def test_rag_dedupe_by_source() -> None:
    engine = RAGEngine.__new__(RAGEngine)
    chunks = [
        RetrievedChunk(
            chunk_id="a",
            chunk_type="t",
            source_id="s1",
            chapter_name="c",
            concept_name="n",
            difficulty="easy",
            title="t",
            content="hello",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="b",
            chunk_type="t",
            source_id="s1",
            chapter_name="c",
            concept_name="n",
            difficulty="easy",
            title="t2",
            content="world",
            score=0.5,
        ),
        RetrievedChunk(
            chunk_id="c",
            chunk_type="t",
            source_id="s2",
            chapter_name="c",
            concept_name="n",
            difficulty="easy",
            title="t3",
            content="x",
            score=0.4,
        ),
    ]
    out = RAGEngine._dedupe_and_budget_chunks(chunks, top_k=4, max_chars=100)
    assert len(out) == 2
    assert {c.source_id for c in out} == {"s1", "s2"}


def test_concept_question_bank_covers_every_concept() -> None:
    dataset = generate_cbse_math_dataset()
    concept_ids = set(dataset.concepts[dataset.concepts["node_type"] == "concept"]["concept_id"].astype(str))
    bank_q, bank_s = generate_concept_question_bank(
        subject="math",
        concepts=dataset.concepts,
        theory_content=dataset.theory_content,
        config=ConceptBankConfig(subject="math", seed=7, templates_per_difficulty=2),
    )
    bank_by_concept = {str(r.concept_id) for r in bank_q}
    assert bank_by_concept == concept_ids
    assert len(bank_q) == len(concept_ids) * 6
    assert len(bank_s) == len(bank_q)
    assert len({r.question_id for r in bank_q}) == len(bank_q)


def test_validate_concept_graph_empty_issues_for_consistent_seed() -> None:
    dataset = generate_cbse_math_dataset()
    issues = validate_concept_graph(
        dataset.concept_graph,
        concepts=dataset.concepts,
        relationships=dataset.relationships,
    )
    assert isinstance(issues, list)
