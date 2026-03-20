from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adaptive_learning.content_generation.generator import (
    GenerationContext,
    build_content_generator,
)
from adaptive_learning.content_generation.io import write_generated_content_bundle


@dataclass(frozen=True)
class ContentGenerationResult:
    questions: pd.DataFrame
    explanations: pd.DataFrame
    summaries: pd.DataFrame
    output_dir: Path


def load_generation_context(*, data_dir: Path) -> GenerationContext:
    return GenerationContext(
        concepts=pd.read_csv(data_dir / "concepts.csv"),
        questions=pd.read_csv(data_dir / "questions.csv"),
        solutions=pd.read_csv(data_dir / "solutions.csv"),
    )


def build_content_bundle(
    *,
    data_dir: Path,
    output_dir: Path,
    concept_ids: list[str],
    backend: str = "grounded",
    model_name: str = "google/flan-t5-base",
) -> ContentGenerationResult:
    context = load_generation_context(data_dir=data_dir)
    generator = build_content_generator(context=context, backend=backend, model_name=model_name)

    generated_questions = []
    generated_explanations = []
    generated_summaries = []

    for concept_id in concept_ids:
        concept = context.concepts[context.concepts["concept_id"] == concept_id].iloc[0]
        difficulty = str(concept.get("difficulty_band", "medium"))
        if difficulty == "core":
            difficulty = "medium"

        question_record, _ = generator.generate_question(concept_id=concept_id, difficulty=difficulty)
        explanation_record, _ = generator.generate_explanation(concept_id=concept_id)
        summary_record, _ = generator.generate_summary(scope_type="concept", scope_id=concept_id)

        generated_questions.append(question_record.to_dict())
        generated_explanations.append(explanation_record.to_dict())
        generated_summaries.append(summary_record.to_dict())

    questions_df = pd.DataFrame(generated_questions)
    explanations_df = pd.DataFrame(generated_explanations)
    summaries_df = pd.DataFrame(generated_summaries)
    write_generated_content_bundle(
        output_dir=output_dir,
        questions=questions_df,
        explanations=explanations_df,
        summaries=summaries_df,
    )
    return ContentGenerationResult(
        questions=questions_df,
        explanations=explanations_df,
        summaries=summaries_df,
        output_dir=output_dir,
    )
