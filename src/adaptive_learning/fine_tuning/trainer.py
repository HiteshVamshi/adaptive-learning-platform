from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class DifficultyCalibrationResult:
    model: Pipeline
    metrics: dict
    label_order: list[str]


def train_difficulty_calibrator(
    *,
    dataset: pd.DataFrame,
    random_seed: int = 7,
) -> DifficultyCalibrationResult:
    train_df, test_df = train_test_split(
        dataset,
        test_size=0.25,
        random_state=random_seed,
        stratify=dataset["difficulty"],
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )
    model.fit(train_df["text"], train_df["difficulty"])

    predictions = model.predict(test_df["text"])
    label_order = model.named_steps["clf"].classes_.tolist()

    metrics = {
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "accuracy": float(accuracy_score(test_df["difficulty"], predictions)),
        "classification_report": classification_report(
            test_df["difficulty"],
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "label_order": label_order,
    }
    return DifficultyCalibrationResult(model=model, metrics=metrics, label_order=label_order)


def save_difficulty_calibrator(
    *,
    output_dir: Path,
    calibration_result: DifficultyCalibrationResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibration_result.model, output_dir / "difficulty_calibrator.joblib")
    with (output_dir / "difficulty_metrics.json").open("w", encoding="utf-8") as file_obj:
        json.dump(calibration_result.metrics, file_obj, indent=2)


def load_difficulty_calibrator(*, input_dir: Path) -> Pipeline:
    return joblib.load(input_dir / "difficulty_calibrator.joblib")
