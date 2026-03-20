from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_recommendation_artifacts(
    *,
    output_dir: Path,
    recommendations: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(output_dir / "recommendations.csv", index=False)

    summary = {
        "recommendation_count": int(len(recommendations)),
        "chapters": recommendations["chapter_name"].value_counts().to_dict(),
        "difficulties": recommendations["difficulty"].value_counts().to_dict(),
        "top_recommendations": recommendations.head(5)[
            ["question_id", "concept_name", "difficulty", "recommendation_score", "rationale"]
        ].to_dict(orient="records"),
    }
    with (output_dir / "recommendation_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
