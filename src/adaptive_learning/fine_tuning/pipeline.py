from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adaptive_learning.fine_tuning.adapter import DifficultyTunedGenerator
from adaptive_learning.fine_tuning.dataset import build_difficulty_training_dataset
from adaptive_learning.fine_tuning.io import write_fine_tuning_bundle
from adaptive_learning.fine_tuning.trainer import (
    save_difficulty_calibrator,
    train_difficulty_calibrator,
)


@dataclass(frozen=True)
class FineTuningPipelineResult:
    dataset: pd.DataFrame
    comparisons: pd.DataFrame
    output_dir: Path
    metrics: dict


def run_fine_tuning_pipeline(
    *,
    data_dir: Path,
    output_dir: Path,
    concept_pairs: list[tuple[str, str]],
    random_seed: int = 7,
) -> FineTuningPipelineResult:
    dataset = build_difficulty_training_dataset(data_dir=data_dir)
    calibration_result = train_difficulty_calibrator(
        dataset=dataset,
        random_seed=random_seed,
    )
    save_difficulty_calibrator(
        output_dir=output_dir,
        calibration_result=calibration_result,
    )

    tuned_generator = DifficultyTunedGenerator.from_artifacts(
        data_dir=data_dir,
        fine_tuning_dir=output_dir,
    )
    comparisons = pd.DataFrame(
        [
            tuned_generator.compare_generation(
                concept_id=concept_id,
                target_difficulty=difficulty,
            ).to_dict()
            for concept_id, difficulty in concept_pairs
        ]
    )

    write_fine_tuning_bundle(
        output_dir=output_dir,
        dataset=dataset,
        comparisons=comparisons,
    )

    return FineTuningPipelineResult(
        dataset=dataset,
        comparisons=comparisons,
        output_dir=output_dir,
        metrics=calibration_result.metrics,
    )
