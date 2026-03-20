from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_generated_content_bundle(
    *,
    output_dir: Path,
    questions: pd.DataFrame,
    explanations: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    questions.to_csv(output_dir / "generated_questions.csv", index=False)
    explanations.to_csv(output_dir / "generated_explanations.csv", index=False)
    summaries.to_csv(output_dir / "generated_summaries.csv", index=False)

    summary = {
        "generated_question_count": int(len(questions)),
        "generated_explanation_count": int(len(explanations)),
        "generated_summary_count": int(len(summaries)),
    }
    with (output_dir / "content_generation_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
