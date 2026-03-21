from __future__ import annotations

import pandas as pd
import streamlit as st

from adaptive_learning.ui.data_access import PlatformArtifacts, PlatformServices
from adaptive_learning.ui.pages._common import subject_label
from adaptive_learning.ui.table_display import column_config_search_results, show_table

__all__ = ["render_search_page"]


def render_search_page(*, artifacts: PlatformArtifacts, services: PlatformServices) -> None:
    st.title("Search")
    st.caption(f"Hybrid retrieval over CBSE Class 10 {subject_label(artifacts.subject)} questions and worked solutions.")

    default_query = "hard trigonometry ratios question"
    query = st.text_input("Search query", value=default_query)
    top_k = st.slider("Top K results", min_value=3, max_value=10, value=5)

    if not query.strip():
        st.info("Enter a query to inspect keyword, vector, and hybrid retrieval.")
        return

    intent, results = services.search_engine.search(query, top_k=top_k)
    result_df = pd.DataFrame(
        [
            {
                "question_id": row.question_id,
                "chapter": row.chapter_name,
                "concept": row.concept_name,
                "difficulty": row.difficulty,
                "bm25_score": round(row.bm25_score, 4),
                "vector_score": round(row.vector_score, 4),
                "hybrid_score": round(row.hybrid_score, 4),
                "exact_concept_match": row.exact_concept_match,
                "concept_overlap": row.concept_overlap,
                "difficulty_match": row.difficulty_match,
                "prompt": row.prompt,
                "final_answer": row.final_answer,
            }
            for row in results
        ]
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Detected Concepts", len(intent.detected_concept_ids))
    metric_cols[1].metric("Expanded Concepts", len(intent.expanded_concept_ids))
    metric_cols[2].metric("Requested Difficulty", intent.requested_difficulty or "none")
    metric_cols[3].metric("Returned Results", len(result_df))

    st.subheader("Query Understanding")
    st.json(
        {
            "normalized_query": intent.normalized_query,
            "detected_concepts": intent.detected_concept_ids,
            "expanded_concepts": intent.expanded_concept_ids,
            "expanded_terms": intent.expanded_terms,
        }
    )

    if result_df.empty:
        st.warning("No results returned for this query.")
        return

    st.subheader("Hybrid Results")
    hdf = result_df[
        [
            "question_id",
            "chapter",
            "concept",
            "difficulty",
            "bm25_score",
            "vector_score",
            "hybrid_score",
            "prompt",
        ]
    ].copy()
    show_table(hdf, column_config=column_config_search_results(hdf), key="search_hybrid", height=420)

    st.subheader("Retrieval Score Comparison")
    score_df = result_df.set_index("question_id")[["bm25_score", "vector_score", "hybrid_score"]]
    st.bar_chart(score_df)

    with st.expander("Answer Keys for Retrieved Results", expanded=False):
        adf = result_df[["question_id", "final_answer"]].copy()
        show_table(
            adf,
            column_config=column_config_search_results(adf),
            key="search_answers",
            height=240,
        )
