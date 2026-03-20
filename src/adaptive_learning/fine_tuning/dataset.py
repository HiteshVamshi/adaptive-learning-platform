from __future__ import annotations

from pathlib import Path

import pandas as pd

from adaptive_learning.content_generation.generator import build_content_generator
from adaptive_learning.content_generation.pipeline import load_generation_context
from adaptive_learning.fine_tuning.schemas import DifficultyExampleRecord


DIFFICULTY_STYLE_HINTS = {
    "easy": "Solve it directly in one clean step.",
    "medium": "Show the main intermediate steps clearly.",
    "hard": "Justify the theorem, identity, or reasoning used at each key step.",
}

DIFFICULTY_STYLE_TAGS = {
    "easy": "direct single-step short computation quick check",
    "medium": "multi-step guided intermediate steps structured working",
    "hard": "proof reasoning justification theorem identity detailed derivation",
}

DIFFICULTY_OBJECTIVES = {
    "easy": "Target difficulty easy. Keep the task short and procedural.",
    "medium": "Target difficulty medium. Require at least one intermediate transformation.",
    "hard": "Target difficulty hard. Require justification, proof, or non-trivial reasoning.",
}


def _compose_training_text(
    *,
    prompt: str,
    answer: str,
    explanation: str,
    concept_name: str,
    chapter_name: str,
    difficulty: str,
    source: str,
    question_type: str,
) -> str:
    style_hint = DIFFICULTY_STYLE_HINTS[difficulty]
    style_tags = DIFFICULTY_STYLE_TAGS[difficulty]
    objective = DIFFICULTY_OBJECTIVES[difficulty]
    return (
        f"Source: {source}. Difficulty label: {difficulty}. {objective} "
        f"Style instruction: {style_hint} Difficulty cues: {style_tags}. "
        f"Concept: {concept_name}. Chapter: {chapter_name}. Question type: {question_type}. "
        f"Prompt: {prompt} Final answer: {answer} Explanation: {explanation}"
    )


def build_difficulty_training_dataset(*, data_dir: Path) -> pd.DataFrame:
    context = load_generation_context(data_dir=data_dir)
    generator = build_content_generator(context=context, backend="grounded")
    solutions = context.solutions.rename(
        columns={
            "final_answer": "solution_final_answer",
            "worked_solution": "solution_worked_solution",
        }
    )
    question_bank = context.questions.merge(solutions, on="question_id", how="left")

    examples: list[DifficultyExampleRecord] = []

    for row in question_bank.to_dict(orient="records"):
        text = _compose_training_text(
            prompt=str(row["prompt"]),
            answer=str(row.get("solution_final_answer") or row["answer"]),
            explanation=str(row.get("solution_worked_solution") or row["answer"]),
            concept_name=str(row["concept_name"]),
            chapter_name=str(row["chapter_name"]),
            difficulty=str(row["difficulty"]),
            source="original_question_bank",
            question_type=str(row["question_type"]),
        )
        examples.append(
            DifficultyExampleRecord(
                example_id=f"orig::{row['question_id']}",
                source="original_question_bank",
                concept_id=str(row["concept_id"]),
                concept_name=str(row["concept_name"]),
                chapter_name=str(row["chapter_name"]),
                difficulty=str(row["difficulty"]),
                text=text,
            )
        )

    concept_rows = context.concepts[context.concepts["node_type"] == "concept"]
    for concept in concept_rows.to_dict(orient="records"):
        concept_id = str(concept["concept_id"])
        concept_name = str(concept["name"])
        chapter_name = str(concept["chapter_name"])

        for difficulty in ["easy", "medium", "hard"]:
            for variant_id in range(2):
                generated_question, _ = generator.generate_question(
                    concept_id=concept_id,
                    difficulty=difficulty,
                )
                prompt = (
                    f"{generated_question.prompt} "
                    f"{DIFFICULTY_OBJECTIVES[difficulty]} "
                    f"Variant {variant_id + 1}."
                )
                text = _compose_training_text(
                    prompt=prompt,
                    answer=generated_question.final_answer,
                    explanation=generated_question.explanation,
                    concept_name=concept_name,
                    chapter_name=chapter_name,
                    difficulty=difficulty,
                    source="grounded_generation",
                    question_type="generated_practice",
                )
                examples.append(
                    DifficultyExampleRecord(
                        example_id=f"generated::{concept_id}::{difficulty}::v{variant_id + 1}",
                        source="grounded_generation",
                        concept_id=concept_id,
                        concept_name=concept_name,
                        chapter_name=chapter_name,
                        difficulty=difficulty,
                        text=text,
                    )
                )

    return pd.DataFrame([example.to_dict() for example in examples])
