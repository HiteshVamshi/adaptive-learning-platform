from __future__ import annotations

import pandas as pd
import streamlit as st

from adaptive_learning.ui.data_access import PlatformArtifacts
from adaptive_learning.ui.pages._common import concept_options, subject_label
from adaptive_learning.ui.table_display import column_config_mastery, show_table
from adaptive_learning.ui.visualization import graph_to_dot

__all__ = ["render_analysis_dashboard"]


def render_analysis_dashboard(
    *,
    artifacts: PlatformArtifacts,
    live_snapshot: pd.DataFrame,
    user_mastery_history: pd.DataFrame | None = None,
) -> None:
    st.title("Analysis Dashboard")
    st.caption(
        f"Mastery, recommendations, and concept-graph structure for the current "
        f"{subject_label(artifacts.subject)} learner state."
    )

    band_counts = live_snapshot["mastery_band"].value_counts()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Concepts Tracked", int(len(live_snapshot)))
    metric_cols[1].metric("Mastered", int(band_counts.get("mastered", 0)))
    metric_cols[2].metric("Developing", int(band_counts.get("developing", 0)))
    metric_cols[3].metric("Needs Support", int(band_counts.get("needs_support", 0)))

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Weakest Concepts")
        weakest = live_snapshot.sort_values(by="graph_adjusted_mastery").head(7)
        wdf = weakest[["concept_name", "chapter_name", "graph_adjusted_mastery", "mastery_band"]].copy()
        show_table(wdf, column_config=column_config_mastery(wdf), key="analysis_weakest", height=280)
    with right_col:
        st.subheader("Strongest Concepts")
        strongest = live_snapshot.sort_values(by="graph_adjusted_mastery", ascending=False).head(7)
        sdf = strongest[["concept_name", "chapter_name", "graph_adjusted_mastery", "mastery_band"]].copy()
        show_table(sdf, column_config=column_config_mastery(sdf), key="analysis_strongest", height=280)

    st.subheader("Mastery by Concept")
    chart_df = live_snapshot.set_index("concept_name")[["graph_adjusted_mastery"]]
    st.bar_chart(chart_df)

    opts = concept_options(artifacts.concepts)
    selected_concept = st.selectbox(
        "Mastery history concept",
        options=opts,
        format_func=lambda item: item[0],
        key="history_concept",
    )
    history_df = (
        user_mastery_history
        if user_mastery_history is not None and not user_mastery_history.empty
        else artifacts.mastery_history
    )
    history_df = history_df[history_df["concept_id"] == selected_concept[1]].copy()
    history_df = history_df.sort_values(by="simulation_step")
    if not history_df.empty:
        st.subheader("Mastery Over Time")
        st.line_chart(
            history_df.set_index("simulation_step")[
                ["direct_mastery", "graph_adjusted_mastery"]
            ]
        )

    st.subheader("Knowledge Graph")
    graph_focus = st.selectbox(
        "Graph focus",
        options=[("Chapter overview", "")] + opts,
        format_func=lambda item: item[0],
        key="graph_focus",
    )
    dot = graph_to_dot(
        artifacts.concept_graph,
        focus_concept_id=graph_focus[1] or None,
    )
    try:
        st.graphviz_chart(dot)
    except Exception:
        st.code(dot, language="dot")
