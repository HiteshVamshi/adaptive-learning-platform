"""User progress dashboard with chapter and topic heatmaps."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from adaptive_learning.ui.data_access import PlatformArtifacts
from adaptive_learning.ui.page_views._common import subject_label
from adaptive_learning.ui.table_display import column_config_mastery, show_table

__all__ = ["render_user_dashboard"]

_PLOTLY_BASE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(247, 245, 239, 0.45)",
    plot_bgcolor="rgba(255, 255, 255, 0.92)",
    font=dict(family="Segoe UI, system-ui, sans-serif", size=12, color="#1c1c1c"),
    title_font=dict(size=15, color="#1f5aa6"),
    margin=dict(l=48, r=28, t=52, b=48),
)


def _polish(fig):
    fig.update_layout(**_PLOTLY_BASE)
    return fig


def _chapter_concept_heatmap(snapshot: pd.DataFrame, artifacts: PlatformArtifacts) -> go.Figure:
    """Chapter × Concept mastery heatmap (one column per concept slot within chapter)."""
    df = snapshot.copy()
    chapters = df["chapter_name"].unique().tolist()
    concepts_per_chapter = (
        df.groupby("chapter_name")["concept_name"]
        .apply(list)
        .to_dict()
    )

    max_concepts = max(len(v) for v in concepts_per_chapter.values()) if concepts_per_chapter else 1
    z_rows = []
    text_rows = []
    hover_rows = []
    for ch in chapters:
        concepts = concepts_per_chapter.get(ch, [])
        z_row, text_row, hover_row = [], [], []
        for i, c in enumerate(concepts):
            val = df[(df["chapter_name"] == ch) & (df["concept_name"] == c)]["graph_adjusted_mastery"]
            v = float(val.iloc[0]) if len(val) else 0.0
            z_row.append(v)
            text_row.append(f"{v:.0%}")
            hover_row.append(c)
        while len(z_row) < max_concepts:
            z_row.append(None)
            text_row.append("")
            hover_row.append("")
        z_rows.append(z_row)
        text_rows.append(text_row)
        hover_rows.append(hover_row)

    col_labels = [f"C{i+1}" for i in range(max_concepts)]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_rows,
            x=col_labels,
            y=chapters,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=text_rows,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            customdata=hover_rows,
            hovertemplate="Chapter: %{y}<br>Concept: %{customdata}<br>Mastery: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Mastery by Chapter and Concept (C1, C2, … = concept order)",
        xaxis_title="Concept",
        yaxis_title="Chapter",
        height=400 + len(chapters) * 28,
    )
    return _polish(fig)


def _chapter_level_heatmap(snapshot: pd.DataFrame) -> go.Figure:
    """Chapter-level aggregate: avg mastery per chapter."""
    agg = (
        snapshot.groupby("chapter_name")["graph_adjusted_mastery"]
        .agg(["mean", "count"])
        .reset_index()
    )
    agg["mastery_pct"] = (agg["mean"] * 100).round(1)
    fig = px.bar(
        agg,
        x="chapter_name",
        y="mean",
        color="mean",
        color_continuous_scale="RdYlGn",
        range_color=(0, 1),
        labels={"mean": "Avg Mastery", "chapter_name": "Chapter"},
        title="Average Mastery by Chapter",
    )
    fig.update_layout(height=400, showlegend=False)
    return _polish(fig)


def _topic_level_heatmap(snapshot: pd.DataFrame, syllabus: pd.DataFrame) -> go.Figure | None:
    """Topic-level mastery: syllabus topics mapped to chapters, then concept mastery averaged."""
    if syllabus.empty or "topic_name" not in syllabus.columns or "chapter_id" not in syllabus.columns:
        return None
    chap = syllabus[["chapter_id", "chapter_name"]].drop_duplicates()
    snap = snapshot.merge(chap, on="chapter_name", how="left")
    if snap["chapter_id"].isna().all():
        return None
    topic_chapters = syllabus.groupby("topic_name")["chapter_id"].apply(set).to_dict()
    rows = []
    for topic, ch_ids in topic_chapters.items():
        subset = snap[snap["chapter_id"].isin(ch_ids)]
        if subset.empty:
            continue
        rows.append({
            "topic": topic,
            "graph_adjusted_mastery": subset["graph_adjusted_mastery"].mean(),
            "concept_count": len(subset),
        })
    if not rows:
        return None
    agg = pd.DataFrame(rows).sort_values("graph_adjusted_mastery", ascending=True)
    fig = px.bar(
        agg,
        x="topic",
        y="graph_adjusted_mastery",
        color="graph_adjusted_mastery",
        color_continuous_scale="RdYlGn",
        range_color=(0, 1),
        labels={"graph_adjusted_mastery": "Mastery", "topic": "Syllabus Topic"},
        title="Mastery by Syllabus Topic",
    )
    fig.update_layout(height=300 + len(agg) * 18, xaxis_tickangle=-45, showlegend=False)
    return _polish(fig)


def render_user_dashboard(
    *,
    artifacts: PlatformArtifacts,
    live_snapshot: pd.DataFrame,
    user_display_name: str,
    total_attempts: int,
) -> None:
    st.title("My Progress")
    st.caption(
        f"Your mastery across {subject_label(artifacts.subject)}: "
        f"chapter heatmap, topic breakdown, and key metrics."
    )

    metric_cols = st.columns(5)
    band_counts = live_snapshot["mastery_band"].value_counts()
    mastered = int(band_counts.get("mastered", 0))
    developing = int(band_counts.get("developing", 0))
    needs = int(band_counts.get("needs_support", 0))

    metric_cols[0].metric("Total Attempts", total_attempts)
    metric_cols[1].metric("Concepts Tracked", int(len(live_snapshot)))
    metric_cols[2].metric("Mastered", mastered)
    metric_cols[3].metric("Developing", developing)
    metric_cols[4].metric("Needs Support", needs)

    st.subheader("Chapter-Level Mastery")
    st.plotly_chart(_chapter_level_heatmap(live_snapshot), use_container_width=True)

    st.subheader("Chapter × Concept Heatmap")
    st.plotly_chart(_chapter_concept_heatmap(live_snapshot, artifacts), use_container_width=True)

    topic_fig = _topic_level_heatmap(live_snapshot, artifacts.syllabus_topics)
    if topic_fig is not None:
        st.subheader("Syllabus Topic Mastery")
        st.plotly_chart(topic_fig, use_container_width=True)

    st.subheader("Weakest Concepts")
    weakest = live_snapshot.sort_values("graph_adjusted_mastery").head(10)
    wdf = weakest[
        ["concept_name", "chapter_name", "graph_adjusted_mastery", "mastery_band", "attempts_count"]
    ].copy()
    show_table(wdf, column_config=column_config_mastery(wdf), key="ud_weakest_concepts", height=380)
