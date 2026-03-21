from __future__ import annotations

import pandas as pd
import streamlit as st

from adaptive_learning.ui.data_access import PlatformArtifacts, PlatformServices
from adaptive_learning.ui.table_display import column_config_debug_scores, column_config_mastery, show_table

__all__ = ["render_debug_page"]


def render_debug_page(
    *,
    artifacts: PlatformArtifacts,
    services: PlatformServices,
    live_snapshot: pd.DataFrame,
) -> None:
    st.title("System Debug View")
    st.caption("Raw internals across retrieval, RAG, agents, mastery, and fine-tuning.")

    debug_query = st.text_input("Debug query", value="Explain irrational numbers proof")
    debug_cols = st.columns(3)
    if debug_query.strip():
        intent, search_results = services.search_engine.search(debug_query, top_k=3)
        rag_response = services.rag_engine.answer_question(debug_query, top_k=4)
        query_agent_response = services.agent_suite.query_agent.respond(debug_query)

        with debug_cols[0]:
            st.subheader("Search Internals")
            st.json(
                {
                    "detected_concepts": intent.detected_concept_ids,
                    "expanded_terms": intent.expanded_terms,
                    "requested_difficulty": intent.requested_difficulty,
                }
            )
            sdf = pd.DataFrame(
                [
                    {
                        "question_id": row.question_id,
                        "concept": row.concept_name,
                        "bm25": row.bm25_score,
                        "vector": row.vector_score,
                        "hybrid": row.hybrid_score,
                    }
                    for row in search_results
                ]
            )
            show_table(sdf, column_config=column_config_debug_scores(sdf), key="dbg_search", height=200)

        with debug_cols[1]:
            st.subheader("RAG Context")
            st.markdown(rag_response.answer.answer)
            rdf = pd.DataFrame(
                [
                    {
                        "chunk_id": row.chunk_id,
                        "type": row.chunk_type,
                        "concept": row.concept_name,
                        "score": row.score,
                    }
                    for row in rag_response.retrieved_chunks
                ]
            )
            show_table(rdf, column_config=column_config_debug_scores(rdf), key="dbg_rag", height=220)

        with debug_cols[2]:
            st.subheader("Agent Trace")
            st.markdown(query_agent_response.answer)
            st.json([trace.to_dict() for trace in query_agent_response.tool_trace])

    st.subheader("Artifact Summaries")
    summary_name = st.selectbox(
        "Summary artifact",
        options=[
            "dataset_summary",
            "search_summary",
            "rag_summary",
            "mastery_summary",
            "recommendation_summary",
            "content_generation_summary",
            "fine_tuning_summary",
            "fine_tuning_metrics",
        ],
    )
    st.json(getattr(artifacts, summary_name))

    st.subheader("Saved Agent Traces")
    if artifacts.agent_traces:
        trace_name = st.selectbox("Trace file", options=sorted(artifacts.agent_traces))
        st.json(artifacts.agent_traces[trace_name])

    st.subheader("Live Mastery Snapshot")
    snap = live_snapshot.sort_values(by="graph_adjusted_mastery").head(10).copy()
    show_table(snap, column_config=column_config_mastery(snap), key="dbg_snapshot", height=360)

    st.subheader("Adaptation Comparisons")
    ac = artifacts.adaptation_comparisons.copy()
    show_table(ac, key="dbg_adapt", height=400)
