from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adaptive_learning.data.cbse_math_seed import (
    chapter_and_concept_seed,
    question_seed,
    relationship_seed,
)
from adaptive_learning.kg.concept_graph import build_concept_graph


@dataclass
class GeneratedDataset:
    concepts: pd.DataFrame
    questions: pd.DataFrame
    solutions: pd.DataFrame
    relationships: pd.DataFrame
    concept_graph: object


def generate_cbse_math_dataset() -> GeneratedDataset:
    concepts = pd.DataFrame(
        [record.to_dict() for record in chapter_and_concept_seed()]
    ).sort_values(by=["order_index", "concept_id"], ignore_index=True)

    question_records, solution_records = question_seed()
    questions = pd.DataFrame([record.to_dict() for record in question_records]).sort_values(
        by=["chapter_id", "concept_id", "difficulty", "question_id"], ignore_index=True
    )
    solutions = pd.DataFrame([record.to_dict() for record in solution_records]).sort_values(
        by=["question_id"], ignore_index=True
    )
    relationships = pd.DataFrame(
        [record.to_dict() for record in relationship_seed()]
    ).sort_values(
        by=["relation_type", "source_concept_id", "target_concept_id"],
        ignore_index=True,
    )

    concept_graph = build_concept_graph(concepts=concepts, relationships=relationships)

    return GeneratedDataset(
        concepts=concepts,
        questions=questions,
        solutions=solutions,
        relationships=relationships,
        concept_graph=concept_graph,
    )
