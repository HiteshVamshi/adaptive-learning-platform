from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from networkx.readwrite import json_graph

from adaptive_learning.mastery.engine import ConceptMasteryEngine
from adaptive_learning.mastery.io import write_mastery_artifacts
from adaptive_learning.mastery.simulation import (
    DEFAULT_STUDENT_PROFILES,
    StudentAttemptSimulator,
    StudentProfile,
)


@dataclass(frozen=True)
class MasteryPipelineResult:
    student_profile: StudentProfile
    attempts: pd.DataFrame
    snapshot: pd.DataFrame
    history: pd.DataFrame
    output_dir: Path


def run_mastery_pipeline(
    *,
    data_dir: Path,
    output_dir: Path,
    user_id: str = "student_cbse_01",
    num_attempts: int = 60,
    random_seed: int = 7,
) -> MasteryPipelineResult:
    concepts = pd.read_csv(data_dir / "concepts.csv")
    questions = pd.read_csv(data_dir / "questions.csv")

    with (data_dir / "concept_graph.json").open("r", encoding="utf-8") as file_obj:
        concept_graph = json_graph.node_link_graph(json.load(file_obj), multigraph=True)

    student_profile = DEFAULT_STUDENT_PROFILES[user_id]
    simulator = StudentAttemptSimulator(questions=questions, random_seed=random_seed)
    attempts = simulator.simulate(
        student_profile=student_profile,
        num_attempts=num_attempts,
    )

    engine = ConceptMasteryEngine(concepts=concepts, concept_graph=concept_graph)
    mastery_result = engine.run(attempts=attempts)

    write_mastery_artifacts(
        output_dir=output_dir,
        student_profile=student_profile,
        attempts=attempts,
        snapshot=mastery_result.snapshot,
        history=mastery_result.history,
    )

    return MasteryPipelineResult(
        student_profile=student_profile,
        attempts=attempts,
        snapshot=mastery_result.snapshot,
        history=mastery_result.history,
        output_dir=output_dir,
    )
