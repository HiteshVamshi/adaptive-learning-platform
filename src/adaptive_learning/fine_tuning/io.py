from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_fine_tuning_bundle(
    *,
    output_dir: Path,
    dataset: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_dir / "difficulty_training_dataset.csv", index=False)
    comparisons.to_csv(output_dir / "adaptation_comparisons.csv", index=False)

    summary = {
        "training_example_count": int(len(dataset)),
        "comparison_count": int(len(comparisons)),
    }
    with (output_dir / "fine_tuning_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
