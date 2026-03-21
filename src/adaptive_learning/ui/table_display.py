"""Richer Streamlit tables: shared CSS and column_config presets."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def inject_app_table_styles() -> None:
    """Call once per app run (e.g. from streamlit_app.main). Improves dataframe chrome and metrics."""
    st.markdown(
        """
        <style>
            [data-testid="stDataFrame"] {
                border-radius: 0.65rem;
                box-shadow: 0 4px 18px rgba(31, 90, 166, 0.08);
                border: 1px solid rgba(31, 90, 166, 0.14);
            }
            [data-testid="stDataFrame"] div[data-testid="stTable"] {
                font-variant-numeric: tabular-nums;
            }
            div[data-testid="stMetricValue"] {
                font-weight: 650;
                letter-spacing: -0.02em;
            }
            div[data-testid="stMetricLabel"] {
                font-size: 0.82rem;
                opacity: 0.92;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pick(columns: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    return {k: v for k, v in columns.items() if k in df.columns}


def column_config_mastery(df: pd.DataFrame) -> dict[str, Any]:
    """Progress bar for mastery, readable bands, counts."""
    cfg: dict[str, Any] = {
        "concept_name": st.column_config.TextColumn("Concept", width="large"),
        "chapter_name": st.column_config.TextColumn("Chapter", width="medium"),
        "graph_adjusted_mastery": st.column_config.ProgressColumn(
            "Mastery",
            help="0–100% from your attempts and the concept graph.",
            format="%.0f%%",
            min_value=0.0,
            max_value=1.0,
        ),
        "mastery_band": st.column_config.TextColumn("Level", width="small"),
        "attempts_count": st.column_config.NumberColumn("Attempts", format="%d", step=1),
        "explanation": st.column_config.TextColumn("Note", width="large"),
        "delta": st.column_config.NumberColumn("Change", format="%.4f", help="Change in mastery after this session."),
    }
    return _pick(cfg, df)


def column_config_learn_queue(df: pd.DataFrame) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "concept_name": st.column_config.TextColumn("Concept", width="medium"),
        "mastery_band": st.column_config.TextColumn("Level", width="small"),
        "days_since_last_attempt": st.column_config.NumberColumn("Days since try", format="%d", min_value=0),
        "priority": st.column_config.NumberColumn(
            "Priority",
            help="Higher = review sooner (mastery gap + time since last attempt).",
            format="%.2f",
        ),
        "reason": st.column_config.TextColumn("Why", width="medium"),
        "suggested_action": st.column_config.TextColumn("Suggestion", width="medium"),
    }
    return _pick(cfg, df)


def column_config_recommendations(df: pd.DataFrame) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "recommendation_rank": st.column_config.NumberColumn("Rank", format="%d", step=1),
        "question_id": st.column_config.TextColumn("Question", width="medium"),
        "concept_name": st.column_config.TextColumn("Concept", width="medium"),
        "difficulty": st.column_config.TextColumn("Difficulty", width="small"),
        "recommendation_score": st.column_config.ProgressColumn(
            "Score",
            format="%.0f%%",
            min_value=0.0,
            max_value=1.0,
        ),
        "rationale": st.column_config.TextColumn("Why", width="large"),
    }
    return _pick(cfg, df)


def column_config_search_results(df: pd.DataFrame) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "question_id": st.column_config.TextColumn("Question ID", width="medium"),
        "chapter": st.column_config.TextColumn("Chapter", width="small"),
        "concept": st.column_config.TextColumn("Concept", width="medium"),
        "difficulty": st.column_config.TextColumn("Difficulty", width="small"),
        "bm25_score": st.column_config.NumberColumn("BM25", format="%.4f"),
        "vector_score": st.column_config.NumberColumn("Vector", format="%.4f"),
        "hybrid_score": st.column_config.ProgressColumn(
            "Hybrid",
            format="%.0f%%",
            min_value=0.0,
            max_value=1.0,
        ),
        "prompt": st.column_config.TextColumn("Prompt", width="large"),
        "final_answer": st.column_config.TextColumn("Answer", width="medium"),
    }
    return _pick(cfg, df)


def column_config_coverage(df: pd.DataFrame) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "chapter_name": st.column_config.TextColumn("Chapter", width="medium"),
        "unit_name": st.column_config.TextColumn("Unit", width="small"),
        "unit_marks": st.column_config.NumberColumn("Marks", format="%d", step=1),
        "periods": st.column_config.NumberColumn("Periods", format="%d", step=1),
        "textbook_section_count": st.column_config.NumberColumn("Sections", format="%d", step=1),
        "concept_count": st.column_config.NumberColumn("Concepts", format="%d", step=1),
        "theory_count": st.column_config.NumberColumn("Theory", format="%d", step=1),
        "question_count": st.column_config.NumberColumn("Questions", format="%d", step=1),
        "question_sources": st.column_config.TextColumn("Sources", width="medium"),
        "coverage_status": st.column_config.TextColumn(
            "Status",
            help="covered = questions + theory; partial = one of them; missing = neither.",
            width="small",
        ),
    }
    return _pick(cfg, df)


def column_config_official_sources(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "label": st.column_config.TextColumn("Source", width="medium"),
            "kind": st.column_config.TextColumn("Type", width="small"),
            "document": st.column_config.TextColumn("Document", width="large"),
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
            "usage": st.column_config.TextColumn("Usage", width="large"),
        },
        df,
    )


def column_config_question_bank(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "question_id": st.column_config.TextColumn("ID", width="medium"),
            "chapter_name": st.column_config.TextColumn("Chapter", width="small"),
            "concept_name": st.column_config.TextColumn("Concept", width="medium"),
            "difficulty": st.column_config.TextColumn("Level", width="small"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "prompt": st.column_config.TextColumn("Prompt", width="large"),
            "answer": st.column_config.TextColumn("Answer", width="medium"),
        },
        df,
    )


def column_config_concepts_compact(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "concept_id": st.column_config.TextColumn("ID", width="medium"),
            "name": st.column_config.TextColumn("Name", width="medium"),
            "node_type": st.column_config.TextColumn("Type", width="small"),
            "chapter_name": st.column_config.TextColumn("Chapter", width="small"),
            "source_kind": st.column_config.TextColumn("Kind", width="small"),
            "source_url": st.column_config.LinkColumn("Source link", display_text="Link"),
        },
        df,
    )


def column_config_debug_scores(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "question_id": st.column_config.TextColumn("Question", width="medium"),
            "concept": st.column_config.TextColumn("Concept", width="medium"),
            "bm25": st.column_config.NumberColumn("BM25", format="%.4f"),
            "vector": st.column_config.NumberColumn("Vector", format="%.4f"),
            "hybrid": st.column_config.NumberColumn("Hybrid", format="%.4f"),
            "chunk_id": st.column_config.TextColumn("Chunk", width="medium"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "score": st.column_config.NumberColumn("Score", format="%.4f"),
        },
        df,
    )


def column_config_syllabus(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "unit_name": st.column_config.TextColumn("Unit", width="small"),
            "chapter_name": st.column_config.TextColumn("Chapter", width="medium"),
            "topic_name": st.column_config.TextColumn("Topic", width="large"),
            "unit_marks": st.column_config.NumberColumn("Marks", format="%d", step=1),
            "periods": st.column_config.NumberColumn("Periods", format="%d", step=1),
            "official_text": st.column_config.TextColumn("Syllabus excerpt", width="large"),
            "source_url": st.column_config.LinkColumn("CBSE link", display_text="Open"),
        },
        df,
    )


def column_config_textbook_sections(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "chapter_number": st.column_config.NumberColumn("Ch.#", format="%d", step=1),
            "chapter_name": st.column_config.TextColumn("Chapter", width="medium"),
            "section_number": st.column_config.TextColumn("§", width="small"),
            "section_title": st.column_config.TextColumn("Section", width="large"),
            "start_page": st.column_config.NumberColumn("Page", format="%d", step=1),
            "chapter_pdf_url": st.column_config.LinkColumn("NCERT PDF", display_text="PDF"),
            "source_url": st.column_config.LinkColumn("Index", display_text="Open"),
        },
        df,
    )


def column_config_theory(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "chapter_name": st.column_config.TextColumn("Chapter", width="small"),
            "concept_name": st.column_config.TextColumn("Concept", width="medium"),
            "title": st.column_config.TextColumn("Title", width="medium"),
            "content_type": st.column_config.TextColumn("Type", width="small"),
            "textbook_sections": st.column_config.TextColumn("Sections", width="medium"),
            "source_url": st.column_config.LinkColumn("Source", display_text="Open"),
        },
        df,
    )


def column_config_chapter_sources(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "chapter_name": st.column_config.TextColumn("Chapter", width="medium"),
            "source": st.column_config.TextColumn("Source", width="medium"),
            "question_count": st.column_config.NumberColumn("Questions", format="%d", step=1),
        },
        df,
    )


def column_config_source_counts(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "source": st.column_config.TextColumn("Source", width="medium"),
            "question_count": st.column_config.NumberColumn("Questions", format="%d", step=1),
        },
        df,
    )


def column_config_theory_preview(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "title": st.column_config.TextColumn("Title", width="medium"),
            "summary_text": st.column_config.TextColumn("Summary", width="large"),
            "official_syllabus_text": st.column_config.TextColumn("Syllabus text", width="large"),
        },
        df,
    )


def column_config_adaptation_comparison(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "variant": st.column_config.TextColumn("Variant", width="small"),
            "predicted_label": st.column_config.TextColumn("Predicted", width="small"),
            "target_probability": st.column_config.NumberColumn("Target p", format="%.3f", help="Calibrator target probability."),
            "prompt": st.column_config.TextColumn("Prompt excerpt", width="large"),
        },
        df,
    )


def column_config_db_tables(df: pd.DataFrame) -> dict[str, Any]:
    return _pick(
        {
            "table_name": st.column_config.TextColumn("Table", width="medium"),
            "row_count": st.column_config.NumberColumn("Rows", format="%d", step=1),
        },
        df,
    )


def show_table(
    df: pd.DataFrame,
    *,
    column_config: dict[str, Any] | None = None,
    key: str | None = None,
    height: int | str | None = 360,
) -> None:
    """Styled dataframe with consistent layout."""
    kwargs: dict[str, Any] = {
        "use_container_width": True,
        "hide_index": True,
    }
    if column_config:
        kwargs["column_config"] = column_config
    if key is not None:
        kwargs["key"] = key
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)
