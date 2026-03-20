from __future__ import annotations

import pandas as pd


def build_rag_chunks(
    *,
    concepts: pd.DataFrame,
    questions: pd.DataFrame,
    solutions: pd.DataFrame,
    theory_content: pd.DataFrame | None = None,
) -> pd.DataFrame:
    concept_rows = concepts[concepts["node_type"] == "concept"].copy()
    theory_chunks = pd.DataFrame(
        {
            "chunk_id": concept_rows["concept_id"].map(lambda value: f"theory::{value}"),
            "chunk_type": "theory",
            "source_id": concept_rows["concept_id"],
            "chapter_id": concept_rows["chapter_id"],
            "chapter_name": concept_rows["chapter_name"],
            "concept_id": concept_rows["concept_id"],
            "concept_name": concept_rows["name"],
            "difficulty": concept_rows["difficulty_band"],
            "title": concept_rows["name"].map(lambda value: f"Theory: {value}"),
            "content": (
                "Concept: "
                + concept_rows["name"].fillna("")
                + ". Description: "
                + concept_rows["description"].fillna("")
                + ". Learning objective: "
                + concept_rows["learning_objective"].fillna("")
                + ". Tags: "
                + concept_rows["tags"].fillna("").str.replace("|", ", ", regex=False)
            ),
        }
    )

    textbook_theory_chunks = pd.DataFrame()
    if theory_content is not None and not theory_content.empty:
        textbook_theory_chunks = pd.DataFrame(
            {
                "chunk_id": theory_content["content_id"],
                "chunk_type": theory_content["content_type"],
                "source_id": theory_content["content_id"],
                "chapter_id": theory_content["chapter_id"],
                "chapter_name": theory_content["chapter_name"],
                "concept_id": theory_content["concept_id"],
                "concept_name": theory_content["concept_name"],
                "difficulty": "medium",
                "title": theory_content["title"],
                "content": (
                    "Summary: "
                    + theory_content["summary_text"].fillna("")
                    + ". Official syllabus: "
                    + theory_content["official_syllabus_text"].fillna("")
                    + ". Textbook sections: "
                    + theory_content["textbook_sections"].fillna("").str.replace("|", ", ", regex=False)
                ),
            }
        )

    question_chunks = pd.DataFrame(
        {
            "chunk_id": questions["question_id"].map(lambda value: f"question::{value}"),
            "chunk_type": "question",
            "source_id": questions["question_id"],
            "chapter_id": questions["chapter_id"],
            "chapter_name": questions["chapter_name"],
            "concept_id": questions["concept_id"],
            "concept_name": questions["concept_name"],
            "difficulty": questions["difficulty"],
            "title": questions["question_id"].map(lambda value: f"Practice Question: {value}"),
            "content": (
                "Question: "
                + questions["prompt"].fillna("")
                + ". Expected final answer: "
                + questions["answer"].fillna("")
                + ". Difficulty: "
                + questions["difficulty"].fillna("")
                + ". Tags: "
                + questions["tags"].fillna("").str.replace("|", ", ", regex=False)
            ),
        }
    )

    solution_rows = questions.merge(solutions, on="question_id", how="left")
    solution_chunks = pd.DataFrame(
        {
            "chunk_id": solution_rows["question_id"].map(lambda value: f"solution::{value}"),
            "chunk_type": "solution",
            "source_id": solution_rows["question_id"],
            "chapter_id": solution_rows["chapter_id"],
            "chapter_name": solution_rows["chapter_name"],
            "concept_id": solution_rows["concept_id"],
            "concept_name": solution_rows["concept_name"],
            "difficulty": solution_rows["difficulty"],
            "title": solution_rows["question_id"].map(lambda value: f"Worked Solution: {value}"),
            "content": (
                "Question: "
                + solution_rows["prompt"].fillna("")
                + ". Worked solution: "
                + solution_rows["worked_solution"].fillna("")
                + ". Final answer: "
                + solution_rows["final_answer"].fillna("")
            ),
        }
    )

    chunks = pd.concat(
        [theory_chunks, textbook_theory_chunks, question_chunks, solution_chunks],
        ignore_index=True,
    )
    chunks["retrieval_text"] = (
        chunks["title"].fillna("")
        + " "
        + chunks["chapter_name"].fillna("")
        + " "
        + chunks["concept_name"].fillna("")
        + " "
        + chunks["difficulty"].fillna("")
        + " "
        + chunks["content"].fillna("")
    )
    return chunks
