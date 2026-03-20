from __future__ import annotations

import json
from pathlib import Path

from networkx.readwrite import json_graph

from adaptive_learning.data.generator import GeneratedDataset


def _dataset_summary(dataset: GeneratedDataset) -> dict:
    chapter_counts = (
        dataset.questions.groupby("chapter_name")["question_id"].count().sort_values(ascending=False)
    )
    difficulty_counts = (
        dataset.questions.groupby("difficulty")["question_id"].count().sort_index()
    )
    return {
        "concept_count": int(len(dataset.concepts)),
        "question_count": int(len(dataset.questions)),
        "solution_count": int(len(dataset.solutions)),
        "relationship_count": int(len(dataset.relationships)),
        "questions_per_chapter": chapter_counts.to_dict(),
        "questions_per_difficulty": difficulty_counts.to_dict(),
    }


def write_dataset(dataset: GeneratedDataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset.concepts.to_csv(output_dir / "concepts.csv", index=False)
    dataset.questions.to_csv(output_dir / "questions.csv", index=False)
    dataset.solutions.to_csv(output_dir / "solutions.csv", index=False)
    dataset.relationships.to_csv(output_dir / "concept_relationships.csv", index=False)

    graph_data = json_graph.node_link_data(dataset.concept_graph)
    with (output_dir / "concept_graph.json").open("w", encoding="utf-8") as file_obj:
        json.dump(graph_data, file_obj, indent=2)

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(_dataset_summary(dataset), file_obj, indent=2)
