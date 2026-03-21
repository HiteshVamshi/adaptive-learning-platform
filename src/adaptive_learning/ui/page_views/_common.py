from __future__ import annotations

import pandas as pd

from adaptive_learning.ui.data_access import PlatformArtifacts


def concept_options(concepts: pd.DataFrame) -> list[tuple[str, str]]:
    concept_rows = concepts[concepts["node_type"] == "concept"].copy()
    concept_rows = concept_rows.sort_values(by=["chapter_name", "order_index", "name"])
    return [
        (f"{row['chapter_name']} / {row['name']}", str(row["concept_id"]))
        for row in concept_rows.to_dict(orient="records")
    ]


def option_index(options: list[tuple[str, str]], target_value: str) -> int:
    for index, (_, value) in enumerate(options):
        if value == target_value:
            return index
    return 0


def subject_label(subject: str) -> str:
    mapping = {"math": "Mathematics", "science": "Science"}
    return mapping.get(str(subject).lower(), str(subject).title())


def question_source_summary(*, artifacts: PlatformArtifacts) -> pd.DataFrame:
    return (
        artifacts.questions.groupby(["chapter_name", "source"])["question_id"]
        .count()
        .reset_index(name="question_count")
        .sort_values(by=["chapter_name", "source"])
    )
