from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adaptive_learning.data.cbse_class10_science_sources import (
    chapter_and_concept_seed as science_chapter_and_concept_seed,
    relationship_seed as science_relationship_seed,
    syllabus_topic_seed as science_syllabus_topic_seed,
    textbook_section_seed as science_textbook_section_seed,
)
from adaptive_learning.data.cbse_class10_syllabus import (
    chapter_and_concept_seed as official_chapter_and_concept_seed,
    relationship_seed as official_relationship_seed,
    syllabus_topic_seed,
)
from adaptive_learning.data.cbse_class10_textbook_index import textbook_section_seed
from adaptive_learning.data.cbse_math_seed import question_seed, relationship_seed as legacy_relationship_seed
from adaptive_learning.data.concept_question_bank import concept_bank_questions_to_frames
from adaptive_learning.data.curriculum_content import build_theory_content
from adaptive_learning.data.science_curriculum_content import (
    build_theory_content as build_science_theory_content,
)
from adaptive_learning.data.science_textbook_bootstrap import (
    generate_science_bootstrap_questions,
)
from adaptive_learning.data.textbook_bootstrap import generate_textbook_bootstrap_questions
from adaptive_learning.data.subject_registry import (
    generate_subject_dataset,
    register_subject,
    supported_subjects,
)
from adaptive_learning.kg.concept_graph import build_concept_graph


@dataclass
class GeneratedDataset:
    subject: str
    concepts: pd.DataFrame
    questions: pd.DataFrame
    solutions: pd.DataFrame
    relationships: pd.DataFrame
    syllabus_topics: pd.DataFrame
    textbook_sections: pd.DataFrame
    theory_content: pd.DataFrame
    concept_graph: object


def generate_cbse_math_dataset() -> GeneratedDataset:
    concepts = pd.DataFrame(
        [record.to_dict() for record in official_chapter_and_concept_seed()]
    ).sort_values(by=["order_index", "concept_id"], ignore_index=True)

    manual_question_records, manual_solution_records = question_seed()
    manual_questions = pd.DataFrame([record.to_dict() for record in manual_question_records]).sort_values(
        by=["chapter_id", "concept_id", "difficulty", "question_id"], ignore_index=True
    )
    manual_solutions = pd.DataFrame([record.to_dict() for record in manual_solution_records]).sort_values(
        by=["question_id"], ignore_index=True
    )

    relationship_rows = [
        *[record.to_dict() for record in official_relationship_seed()],
        *[record.to_dict() for record in legacy_relationship_seed()],
    ]
    relationships = (
        pd.DataFrame(relationship_rows)
        .drop_duplicates(
            subset=["source_concept_id", "target_concept_id", "relation_type"],
            keep="first",
        )
        .sort_values(
            by=["relation_type", "source_concept_id", "target_concept_id"],
            ignore_index=True,
        )
    )

    syllabus_topics = pd.DataFrame([record.to_dict() for record in syllabus_topic_seed()]).sort_values(
        by=["unit_id", "chapter_id", "topic_id"], ignore_index=True
    )
    textbook_sections = pd.DataFrame([record.to_dict() for record in textbook_section_seed()]).sort_values(
        by=["chapter_number", "start_page", "section_number"], ignore_index=True
    )
    theory_content = pd.DataFrame(
        build_theory_content(
            concepts_df=concepts,
            syllabus_topics_df=syllabus_topics,
            textbook_sections_df=textbook_sections,
        )
    ).sort_values(by=["chapter_id", "concept_id"], ignore_index=True)

    generated_question_records, generated_solution_records = generate_textbook_bootstrap_questions(
        concepts=concepts,
        existing_questions=manual_questions,
        theory_content=theory_content,
    )
    textbook_questions = pd.DataFrame([record.to_dict() for record in generated_question_records])
    textbook_solutions = pd.DataFrame([record.to_dict() for record in generated_solution_records])

    bank_q, bank_s = concept_bank_questions_to_frames(
        subject="math",
        concepts=concepts,
        theory_content=theory_content,
    )

    questions = pd.concat([manual_questions, textbook_questions, bank_q], ignore_index=True).sort_values(
        by=["chapter_id", "concept_id", "difficulty", "question_id"], ignore_index=True
    )
    solutions = pd.concat([manual_solutions, textbook_solutions, bank_s], ignore_index=True).sort_values(
        by=["question_id"], ignore_index=True
    )

    concept_graph = build_concept_graph(
        concepts=concepts, relationships=relationships, subject_key="math"
    )

    return GeneratedDataset(
        subject="math",
        concepts=concepts,
        questions=questions,
        solutions=solutions,
        relationships=relationships,
        syllabus_topics=syllabus_topics,
        textbook_sections=textbook_sections,
        theory_content=theory_content,
        concept_graph=concept_graph,
    )


def generate_cbse_science_dataset() -> GeneratedDataset:
    concepts = pd.DataFrame(
        [record.to_dict() for record in science_chapter_and_concept_seed()]
    ).sort_values(by=["order_index", "concept_id"], ignore_index=True)

    relationships = pd.DataFrame(
        [record.to_dict() for record in science_relationship_seed()]
    ).sort_values(
        by=["relation_type", "source_concept_id", "target_concept_id"],
        ignore_index=True,
    )

    syllabus_topics = pd.DataFrame(
        [record.to_dict() for record in science_syllabus_topic_seed()]
    ).sort_values(by=["unit_id", "chapter_id", "topic_id"], ignore_index=True)
    textbook_sections = pd.DataFrame(
        [record.to_dict() for record in science_textbook_section_seed()]
    ).sort_values(by=["chapter_number", "start_page", "section_number"], ignore_index=True)
    theory_content = pd.DataFrame(
        build_science_theory_content(
            concepts_df=concepts,
            syllabus_topics_df=syllabus_topics,
            textbook_sections_df=textbook_sections,
        )
    ).sort_values(by=["chapter_id", "concept_id"], ignore_index=True)

    generated_question_records, generated_solution_records = generate_science_bootstrap_questions(
        concepts=concepts,
        theory_content=theory_content,
    )
    bootstrap_questions = pd.DataFrame([record.to_dict() for record in generated_question_records]).sort_values(
        by=["chapter_id", "concept_id", "difficulty", "question_id"],
        ignore_index=True,
    )
    bootstrap_solutions = pd.DataFrame([record.to_dict() for record in generated_solution_records]).sort_values(
        by=["question_id"],
        ignore_index=True,
    )
    bank_q, bank_s = concept_bank_questions_to_frames(
        subject="science",
        concepts=concepts,
        theory_content=theory_content,
    )
    questions = pd.concat([bootstrap_questions, bank_q], ignore_index=True).sort_values(
        by=["chapter_id", "concept_id", "difficulty", "question_id"],
        ignore_index=True,
    )
    solutions = pd.concat([bootstrap_solutions, bank_s], ignore_index=True).sort_values(
        by=["question_id"],
        ignore_index=True,
    )
    concept_graph = build_concept_graph(
        concepts=concepts, relationships=relationships, subject_key="science"
    )

    return GeneratedDataset(
        subject="science",
        concepts=concepts,
        questions=questions,
        solutions=solutions,
        relationships=relationships,
        syllabus_topics=syllabus_topics,
        textbook_sections=textbook_sections,
        theory_content=theory_content,
        concept_graph=concept_graph,
    )


register_subject("math", generate_cbse_math_dataset)
register_subject("science", generate_cbse_science_dataset)
SUPPORTED_SUBJECTS: tuple[str, ...] = supported_subjects()
