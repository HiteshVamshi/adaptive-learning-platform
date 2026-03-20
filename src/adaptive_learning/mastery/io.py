from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adaptive_learning.mastery.simulation import StudentProfile


def write_mastery_artifacts(
    *,
    output_dir: Path,
    student_profile: StudentProfile,
    attempts: pd.DataFrame,
    snapshot: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts.to_csv(output_dir / "attempts.csv", index=False)
    snapshot.to_csv(output_dir / "mastery_snapshot.csv", index=False)
    history.to_csv(output_dir / "mastery_history.csv", index=False)

    summary = {
        "user_id": student_profile.user_id,
        "display_name": student_profile.display_name,
        "attempt_count": int(len(attempts)),
        "mastered_concepts": int((snapshot["mastery_band"] == "mastered").sum()),
        "developing_concepts": int((snapshot["mastery_band"] == "developing").sum()),
        "needs_support_concepts": int((snapshot["mastery_band"] == "needs_support").sum()),
        "lowest_mastery_concepts": snapshot.sort_values(
            by="graph_adjusted_mastery", ascending=True
        )
        .head(5)[["concept_id", "concept_name", "graph_adjusted_mastery"]]
        .to_dict(orient="records"),
    }
    with (output_dir / "mastery_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
