from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adaptive_learning.content_generation.generator import build_content_generator
from adaptive_learning.content_generation.pipeline import load_generation_context
from adaptive_learning.fine_tuning.schemas import AdaptationComparisonRecord
from adaptive_learning.fine_tuning.trainer import load_difficulty_calibrator


ADAPTATION_HINTS = {
    "easy": " Solve it directly in one clean step.",
    "medium": " Show the main intermediate steps clearly.",
    "hard": " Justify the theorem, identity, or reasoning used at each key step.",
}


@dataclass
class DifficultyTunedGenerator:
    calibrator: object
    content_generator: object
    label_order: list[str]

    @classmethod
    def from_artifacts(cls, *, data_dir: Path, fine_tuning_dir: Path) -> "DifficultyTunedGenerator":
        context = load_generation_context(data_dir=data_dir)
        calibrator = load_difficulty_calibrator(input_dir=fine_tuning_dir)
        content_generator = build_content_generator(context=context, backend="grounded")
        label_order = calibrator.named_steps["clf"].classes_.tolist()
        return cls(
            calibrator=calibrator,
            content_generator=content_generator,
            label_order=label_order,
        )

    def compare_generation(self, *, concept_id: str, target_difficulty: str) -> AdaptationComparisonRecord:
        baseline_record, _ = self.content_generator.generate_question(
            concept_id=concept_id,
            difficulty=target_difficulty,
        )

        baseline_text = self._candidate_text(
            prompt=baseline_record.prompt,
            explanation=baseline_record.explanation,
        )
        baseline_prediction, baseline_probability = self._predict_target_alignment(
            baseline_text,
            target_difficulty,
        )

        candidates = []
        for difficulty in ["easy", "medium", "hard"]:
            generated_record, _ = self.content_generator.generate_question(
                concept_id=concept_id,
                difficulty=difficulty,
            )
            augmented_prompt = generated_record.prompt + ADAPTATION_HINTS[target_difficulty]
            candidate_text = self._candidate_text(
                prompt=augmented_prompt,
                explanation=generated_record.explanation,
            )
            predicted_label, target_probability = self._predict_target_alignment(
                candidate_text,
                target_difficulty,
            )
            candidates.append(
                {
                    "record": generated_record,
                    "augmented_prompt": augmented_prompt,
                    "predicted_label": predicted_label,
                    "target_probability": target_probability,
                    "source_difficulty": difficulty,
                }
            )

        best_candidate = max(
            candidates,
            key=lambda item: (item["target_probability"], item["predicted_label"] == target_difficulty),
        )

        concept_name = baseline_record.concept_name
        notes = (
            f"Baseline used the raw {target_difficulty} request. "
            f"Adapted generation selected a {best_candidate['source_difficulty']} candidate and applied a "
            f"difficulty-control hint tuned toward {target_difficulty}."
        )

        return AdaptationComparisonRecord(
            concept_id=concept_id,
            concept_name=concept_name,
            target_difficulty=target_difficulty,
            baseline_prompt=baseline_record.prompt,
            baseline_prediction=baseline_prediction,
            baseline_target_probability=round(float(baseline_probability), 4),
            adapted_prompt=best_candidate["augmented_prompt"],
            adapted_prediction=best_candidate["predicted_label"],
            adapted_target_probability=round(float(best_candidate["target_probability"]), 4),
            adaptation_notes=notes,
        )

    def _predict_target_alignment(self, text: str, target_difficulty: str) -> tuple[str, float]:
        predicted_label = str(self.calibrator.predict([text])[0])
        probabilities = self.calibrator.predict_proba([text])[0]
        probability_by_label = dict(zip(self.label_order, probabilities, strict=False))
        return predicted_label, float(probability_by_label[target_difficulty])

    @staticmethod
    def _candidate_text(*, prompt: str, explanation: str) -> str:
        return f"{prompt} Explanation: {explanation}"
