from __future__ import annotations


def question_generation_prompt(*, concept_name: str, chapter_name: str, difficulty: str) -> str:
    return (
        "Generate one CBSE Class 10 mathematics practice question.\n"
        f"Chapter: {chapter_name}\n"
        f"Concept: {concept_name}\n"
        f"Difficulty: {difficulty}\n"
        "Return the question, final answer, and a stepwise explanation."
    )


def explanation_generation_prompt(*, concept_name: str, audience_level: str) -> str:
    return (
        "Explain a CBSE Class 10 mathematics concept.\n"
        f"Concept: {concept_name}\n"
        f"Audience: {audience_level}\n"
        "Return a concise explanation and the key points to remember."
    )


def summary_generation_prompt(*, title: str) -> str:
    return (
        "Summarize the following CBSE Class 10 mathematics topic.\n"
        f"Topic: {title}\n"
        "Return a short summary and 3-5 bullets of what a student should remember."
    )
