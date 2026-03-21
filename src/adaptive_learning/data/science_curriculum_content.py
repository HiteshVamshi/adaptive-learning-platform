from __future__ import annotations

from adaptive_learning.data.cbse_class10_science_sources import (
    CBSE_SCIENCE_SOURCE_DOCUMENT,
    CBSE_SCIENCE_SOURCE_URL,
)
from adaptive_learning.data.schemas import TheoryContentRecord


def build_theory_content(*, concepts_df, syllabus_topics_df, textbook_sections_df):
    chapter_topic_summary = (
        syllabus_topics_df.groupby(["chapter_id", "chapter_name"])["official_text"]
        .apply(lambda values: " ".join(dict.fromkeys(str(value) for value in values)))
        .to_dict()
    )

    theory_rows = []
    concept_rows = concepts_df[concepts_df["node_type"] == "concept"].copy()
    for concept in concept_rows.to_dict(orient="records"):
        chapter_key = (concept["chapter_id"], concept["chapter_name"])
        official_text = chapter_topic_summary.get(chapter_key, "")
        matched_sections = textbook_sections_df[
            textbook_sections_df["chapter_id"] == str(concept["chapter_id"])
        ].to_dict(orient="records")
        section_titles = [str(row["section_title"]) for row in matched_sections]
        section_ids = [str(row["section_id"]) for row in matched_sections]
        section_phrase = ", ".join(section_titles) if section_titles else str(concept["chapter_name"])
        summary_text = (
            f"{concept['name']} belongs to the NCERT Class X Science chapter {concept['chapter_name']}. "
            f"The official syllabus expects students to {str(concept['learning_objective']).rstrip('.').lower()}. "
            f"In the NCERT textbook, this aligns with {section_phrase}. "
            f"{official_text} "
            f"Core concept description: {concept['description']}"
        )
        theory_rows.append(
            TheoryContentRecord(
                content_id=f"curriculum_note::{concept['concept_id']}",
                class_level=str(concept["class_level"]),
                chapter_id=str(concept["chapter_id"]),
                chapter_name=str(concept["chapter_name"]),
                concept_id=str(concept["concept_id"]),
                concept_name=str(concept["name"]),
                title=f"Theory Note: {concept['name']}",
                content_type="syllabus_textbook_grounded",
                summary_text=summary_text,
                official_syllabus_text=official_text,
                textbook_sections=section_titles,
                textbook_section_ids=section_ids,
                source_document=CBSE_SCIENCE_SOURCE_DOCUMENT,
                source_url=CBSE_SCIENCE_SOURCE_URL,
            ).to_dict()
        )
    return theory_rows
