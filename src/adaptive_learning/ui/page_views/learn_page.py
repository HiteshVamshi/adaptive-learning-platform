from __future__ import annotations

import pandas as pd
import streamlit as st

from adaptive_learning.learn.io import load_learn_resources
from adaptive_learning.learn.selector import select_learn_plan
from adaptive_learning.scheduling.spaced_repetition import build_spaced_repetition_queue, top_review_concepts
from adaptive_learning.ui.data_access import PlatformArtifacts, PlatformPaths
from adaptive_learning.ui.table_display import column_config_learn_queue, show_table


def _youtube_watch_url(url: str) -> str | None:
    if "youtube.com/embed/" not in url:
        return None
    vid = url.split("youtube.com/embed/")[-1].split("?")[0].strip()
    if not vid:
        return None
    return f"https://www.youtube.com/watch?v={vid}"


def render_learn_page(
    *,
    artifacts: PlatformArtifacts,
    live_snapshot: pd.DataFrame,
    user_attempts: pd.DataFrame,
    paths: PlatformPaths,
) -> None:
    st.title("Learn")
    st.caption(
        "NCERT chapter PDFs plus optional YouTube embeds or search links (built from your subject’s syllabus). "
        "Recommendations adapt to mastery: more when you need support, fewer when you are strong."
    )

    resources_df = load_learn_resources(paths.artifacts_dir)
    if resources_df.empty:
        subj = paths.subject
        st.warning(
            f"No learn catalog found. From the project root run: "
            f"`python scripts/build_learn_index.py --subject {subj}` or "
            f"`python scripts/build_all.py --subject {subj}` "
            f"(or `python scripts/generate_data.py --subject {subj}`)."
        )
        return

    st.subheader("Review queue (spaced repetition)")
    queue = build_spaced_repetition_queue(
        user_attempts=user_attempts,
        live_snapshot=live_snapshot,
    )
    if queue.empty:
        st.info("No concepts in your snapshot yet.")
    else:
        qdf = top_review_concepts(queue, top_k=12)[
            [
                "concept_name",
                "mastery_band",
                "days_since_last_attempt",
                "priority",
                "reason",
                "suggested_action",
            ]
        ].copy()
        show_table(qdf, column_config=column_config_learn_queue(qdf), key="learn_review_queue", height=400)

    st.subheader("Recommended for you")
    plan = select_learn_plan(
        live_snapshot=live_snapshot,
        resources_df=resources_df,
        concepts=artifacts.concepts,
        max_concepts=10,
    )
    if not plan:
        st.info("No matching curated resources for your weakest concepts yet. Use the catalog below.")

    for concept_row, res_df in plan:
        band = concept_row.get("mastery_band", "")
        name = concept_row.get("concept_name", concept_row.get("concept_id"))
        with st.expander(f"{name} — {band}", expanded=False):
            st.caption(
                f"Mastery score: {float(concept_row.get('graph_adjusted_mastery', 0)):.2f} · "
                f"Attempts: {int(concept_row.get('attempts_count', 0))}"
            )
            for _, r in res_df.iterrows():
                title = str(r["title"])
                url = str(r["url"])
                rtype = str(r["resource_type"])
                if rtype == "video" and "youtube.com/embed" in url:
                    st.markdown(f"**{title}** ({r['source']})")
                    watch = _youtube_watch_url(url)
                    if watch:
                        st.video(watch)
                    else:
                        st.markdown(f"[Open video]({url})")
                else:
                    st.markdown(f"- **[{rtype}] {title}** — [{url}]({url})")

    st.subheader("Browse catalog by chapter")
    chapters = sorted(resources_df["chapter_name"].astype(str).unique().tolist())
    ch = st.selectbox("Chapter", options=chapters)
    if ch:
        sub = resources_df[resources_df["chapter_name"].astype(str) == ch].sort_values(
            by=["order_index", "resource_id"]
        )
        for _, r in sub.iterrows():
            st.markdown(f"- **{r['title']}** ({r['resource_type']}) — [{r['url']}]({r['url']})")
