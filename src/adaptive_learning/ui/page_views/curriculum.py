from __future__ import annotations

import pandas as pd
import streamlit as st

from adaptive_learning.ui.data_access import PlatformArtifacts, build_chapter_coverage_table
from adaptive_learning.ui.page_views._common import question_source_summary
from adaptive_learning.ui.table_display import (
    column_config_chapter_sources,
    column_config_concepts_compact,
    column_config_coverage,
    column_config_db_tables,
    column_config_official_sources,
    column_config_question_bank,
    column_config_source_counts,
    column_config_syllabus,
    column_config_textbook_sections,
    column_config_theory,
    column_config_theory_preview,
    show_table,
)

__all__ = ["render_curriculum_page"]


def render_curriculum_page(*, artifacts: PlatformArtifacts) -> None:
    st.title("Curriculum & Data")
    st.caption(
        "Official CBSE syllabus and NCERT textbook structure used to build the local curriculum, "
        "knowledge graph, theory notes, and question bank."
    )

    summary = artifacts.dataset_summary
    metric_cols = st.columns(6)
    metric_cols[0].metric("Official Chapters", int(summary.get("official_chapter_count", 0)))
    metric_cols[1].metric("Syllabus Topics", int(summary.get("official_topic_count", 0)))
    metric_cols[2].metric("Textbook Sections", int(summary.get("textbook_section_count", 0)))
    metric_cols[3].metric("Theory Notes", int(summary.get("theory_content_count", 0)))
    metric_cols[4].metric("Questions", int(summary.get("question_count", 0)))
    metric_cols[5].metric("Solutions", int(summary.get("solution_count", 0)))

    st.subheader("Official Sources")
    sources_df = pd.DataFrame(artifacts.official_sources)
    show_table(sources_df, column_config=column_config_official_sources(sources_df), key="curr_sources", height=220)

    coverage_df = build_chapter_coverage_table(artifacts=artifacts)
    st.subheader("Coverage by Official Chapter")
    cov = coverage_df[
        [
            "chapter_name",
            "unit_name",
            "unit_marks",
            "periods",
            "textbook_section_count",
            "concept_count",
            "theory_count",
            "question_count",
            "question_sources",
            "coverage_status",
        ]
    ].copy()
    show_table(cov, column_config=column_config_coverage(cov), key="curr_coverage", height=420)

    chart_df = coverage_df.set_index("chapter_name")[["question_count", "theory_count"]]
    st.bar_chart(chart_df)

    syllabus_tab, textbook_tab, theory_tab, data_tab, db_tab = st.tabs(
        ["Syllabus", "Textbook Index", "Theory Content", "Actual Data", "SQLite DB"]
    )

    with syllabus_tab:
        unit_options = ["All units"] + sorted(artifacts.syllabus_topics["unit_name"].dropna().unique().tolist())
        selected_unit = st.selectbox("Filter syllabus by unit", options=unit_options, key="syllabus_unit")
        syllabus_df = artifacts.syllabus_topics.copy()
        if selected_unit != "All units":
            syllabus_df = syllabus_df[syllabus_df["unit_name"] == selected_unit]
        sdf = syllabus_df[
            [
                "unit_name",
                "chapter_name",
                "topic_name",
                "unit_marks",
                "periods",
                "official_text",
                "source_url",
            ]
        ].copy()
        show_table(sdf, column_config=column_config_syllabus(sdf), key="curr_syllabus", height=400)

    with textbook_tab:
        chapter_options = ["All chapters"] + sorted(
            artifacts.textbook_sections["chapter_name"].dropna().unique().tolist()
        )
        selected_chapter = st.selectbox(
            "Filter textbook sections by chapter",
            options=chapter_options,
            key="textbook_chapter",
        )
        textbook_df = artifacts.textbook_sections.copy()
        if selected_chapter != "All chapters":
            textbook_df = textbook_df[textbook_df["chapter_name"] == selected_chapter]
        tdf = textbook_df[
            [
                "chapter_number",
                "chapter_name",
                "section_number",
                "section_title",
                "start_page",
                "chapter_pdf_url",
                "source_url",
            ]
        ].copy()
        show_table(tdf, column_config=column_config_textbook_sections(tdf), key="curr_textbook", height=400)

    with theory_tab:
        theory_chapter = st.selectbox(
            "Filter theory notes by chapter",
            options=["All chapters"] + sorted(artifacts.theory_content["chapter_name"].dropna().unique().tolist()),
            key="theory_chapter",
        )
        theory_df = artifacts.theory_content.copy()
        if theory_chapter != "All chapters":
            theory_df = theory_df[theory_df["chapter_name"] == theory_chapter]
        thdf = theory_df[
            [
                "chapter_name",
                "concept_name",
                "title",
                "content_type",
                "textbook_sections",
                "source_url",
            ]
        ].copy()
        show_table(thdf, column_config=column_config_theory(thdf), key="curr_theory", height=400)

        with st.expander("Theory Note Preview"):
            preview_df = theory_df[["title", "summary_text", "official_syllabus_text"]].head(5).copy()
            show_table(
                preview_df,
                column_config=column_config_theory_preview(preview_df),
                key="curr_theory_preview",
                height=260,
            )

    with data_tab:
        source_counts = (
            artifacts.questions["source"].value_counts().rename_axis("source").reset_index(name="question_count")
        )
        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("Question Source Mix")
            show_table(
                source_counts,
                column_config=column_config_source_counts(source_counts),
                key="curr_source_counts",
                height=240,
            )
        with right_col:
            st.subheader("Question Coverage by Chapter")
            qsrc = question_source_summary(artifacts=artifacts)
            show_table(
                qsrc,
                column_config=column_config_chapter_sources(qsrc),
                key="curr_q_source",
                height=320,
            )

        chapter_filter = st.selectbox(
            "Filter question bank by chapter",
            options=["All chapters"] + sorted(artifacts.questions["chapter_name"].dropna().unique().tolist()),
            key="question_bank_chapter",
        )
        source_filter = st.selectbox(
            "Filter question bank by source",
            options=["All sources"] + sorted(artifacts.questions["source"].dropna().unique().tolist()),
            key="question_bank_source",
        )
        difficulty_filter = st.selectbox(
            "Filter question bank by difficulty",
            options=["All difficulties", "easy", "medium", "hard"],
            key="question_bank_difficulty",
        )

        questions_df = artifacts.questions.copy()
        if chapter_filter != "All chapters":
            questions_df = questions_df[questions_df["chapter_name"] == chapter_filter]
        if source_filter != "All sources":
            questions_df = questions_df[questions_df["source"] == source_filter]
        if difficulty_filter != "All difficulties":
            questions_df = questions_df[questions_df["difficulty"] == difficulty_filter]

        qdf = questions_df[
            [
                "question_id",
                "chapter_name",
                "concept_name",
                "difficulty",
                "source",
                "prompt",
                "answer",
            ]
        ].copy()
        show_table(qdf, column_config=column_config_question_bank(qdf), key="curr_questions", height=480)

        with st.expander("Concept Table"):
            concept_df = artifacts.concepts.copy()
            cdf = concept_df[
                [
                    "concept_id",
                    "name",
                    "node_type",
                    "chapter_name",
                    "source_kind",
                    "source_url",
                ]
            ].copy()
            show_table(cdf, column_config=column_config_concepts_compact(cdf), key="curr_concepts", height=400)

    with db_tab:
        db_status = artifacts.db_status
        st.subheader("SQLite Mirror Status")
        st.code(db_status["sqlite_path"])
        if not db_status["available"]:
            st.warning("SQLite mirror not found. Re-run generate_data.py with --sqlite-path.")
        else:
            db_cols = st.columns(3)
            db_cols[0].metric("Tables", int(len(db_status["table_counts"])))
            db_cols[1].metric(
                "Rows in questions",
                int(db_status["table_counts"].get("questions", 0)),
            )
            db_cols[2].metric(
                "Rows in theory_content",
                int(db_status["table_counts"].get("theory_content", 0)),
            )
            dbdf = pd.DataFrame(
                [
                    {"table_name": table_name, "row_count": row_count}
                    for table_name, row_count in db_status["table_counts"].items()
                ]
            )
            show_table(dbdf, column_config=column_config_db_tables(dbdf), key="curr_db_tables", height=320)
            st.subheader("Stored Dataset Metadata")
            st.json(db_status["metadata"])
