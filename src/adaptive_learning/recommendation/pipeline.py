from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from networkx.readwrite import json_graph

from adaptive_learning.recommendation.engine import RecommendationEngine
from adaptive_learning.recommendation.io import write_recommendation_artifacts


@dataclass(frozen=True)
class RecommendationPipelineResult:
    recommendations: pd.DataFrame
    output_dir: Path


def run_recommendation_pipeline(
    *,
    data_dir: Path,
    mastery_dir: Path,
    output_dir: Path,
    top_k: int = 10,
) -> RecommendationPipelineResult:
    questions = pd.read_csv(data_dir / "questions.csv")
    mastery_snapshot = pd.read_csv(mastery_dir / "mastery_snapshot.csv")
    attempts = pd.read_csv(mastery_dir / "attempts.csv")

    with (data_dir / "concept_graph.json").open("r", encoding="utf-8") as file_obj:
        concept_graph = json_graph.node_link_graph(json.load(file_obj), multigraph=True)

    engine = RecommendationEngine(
        questions=questions,
        mastery_snapshot=mastery_snapshot,
        attempts=attempts,
        concept_graph=concept_graph,
    )
    recommendations = engine.recommend(top_k=top_k)
    write_recommendation_artifacts(output_dir=output_dir, recommendations=recommendations)

    return RecommendationPipelineResult(
        recommendations=recommendations,
        output_dir=output_dir,
    )
