from __future__ import annotations

import pandas as pd
import streamlit as st

from adaptive_learning.ui.data_access import (
    PlatformArtifacts,
    PlatformServices,
    build_empty_mastery_snapshot,
    build_manual_attempts,
    compute_live_mastery_snapshot,
)
from adaptive_learning.ui.page_views._common import concept_options
from adaptive_learning.ui.table_display import column_config_mastery, show_table
from adaptive_learning.ui.page_views.session import session_keys
from adaptive_learning.ui.user_progress import append_attempts, get_user_attempts, get_next_simulation_step

__all__ = ["render_test_page"]


def render_test_page(
    *,
    artifacts: PlatformArtifacts,
    services: PlatformServices,
    user_id: str,
    paths,
) -> None:
    st.title("Test")
    st.caption("Run a short quiz, record outcomes. Mastery updates live and persists.")

    opts = concept_options(artifacts.concepts)
    chapter_options = sorted(
        {
            (str(row["chapter_name"]), str(row["chapter_id"]))
            for row in artifacts.questions[["chapter_name", "chapter_id"]].to_dict(orient="records")
        }
    )

    control_cols = st.columns(4)
    chapter_choice = control_cols[0].selectbox(
        "Chapter",
        options=[("All chapters", "")] + chapter_options,
        format_func=lambda item: item[0],
    )
    concept_choice = control_cols[1].selectbox(
        "Concept",
        options=[("All concepts", "")] + opts,
        format_func=lambda item: item[0],
    )
    difficulty = control_cols[2].selectbox("Difficulty", options=["all", "easy", "medium", "hard"])
    num_questions = control_cols[3].slider("Questions", min_value=1, max_value=5, value=3)

    question_pool = artifacts.questions.copy()
    if chapter_choice[1]:
        question_pool = question_pool[question_pool["chapter_id"] == chapter_choice[1]]
    if concept_choice[1]:
        question_pool = question_pool[question_pool["concept_id"] == concept_choice[1]]
    if difficulty != "all":
        question_pool = question_pool[question_pool["difficulty"] == difficulty]

    if question_pool.empty:
        st.warning("No questions match the selected filters.")
        return

    seed = st.session_state.get(session_keys.test_seed_key, 13)
    sample_size = min(num_questions, len(question_pool))
    sampled_questions = question_pool.sample(
        n=sample_size,
        random_state=seed,
        replace=False,
    ).sort_values(by=["chapter_name", "concept_name", "difficulty", "question_id"])

    st.button(
        "Shuffle Test Set",
        on_click=lambda: st.session_state.update({session_keys.test_seed_key: seed + 1}),
    )

    with st.form("adaptive_test_form"):
        st.subheader("Question Set")
        evaluations = []
        for row in sampled_questions.to_dict(orient="records"):
            st.markdown(
                f"**{row['question_id']}** [{row['difficulty']}] {row['concept_name']}  \n{row['prompt']}"
            )
            correctness = st.radio(
                f"Result for {row['question_id']}",
                options=["Correct", "Incorrect"],
                horizontal=True,
                key=f"result_{row['question_id']}",
            )
            response_time = st.number_input(
                f"Response time for {row['question_id']} (sec)",
                min_value=20,
                max_value=1200,
                value=int(row["estimated_time_sec"]),
                step=10,
                key=f"time_{row['question_id']}",
            )
            with st.expander(f"Show official answer for {row['question_id']}"):
                solution_row = artifacts.solutions[artifacts.solutions["question_id"] == row["question_id"]].iloc[0]
                st.markdown(f"**Final answer**  \n{solution_row['final_answer']}")
                st.markdown(f"**Worked solution**  \n{solution_row['worked_solution']}")
            evaluations.append(
                {
                    "question_id": row["question_id"],
                    "correct": correctness == "Correct",
                    "response_time_sec": int(response_time),
                }
            )

        submitted = st.form_submit_button("Submit Test Attempts")

    if not submitted:
        return

    prior_attempts = get_user_attempts(paths, artifacts.subject, user_id)
    fallback = build_empty_mastery_snapshot(artifacts.concepts, artifacts.mastery_snapshot)
    before_snapshot = compute_live_mastery_snapshot(
        base_attempts=pd.DataFrame(),
        manual_attempts=prior_attempts,
        mastery_engine=services.mastery_engine,
        fallback_snapshot=fallback,
    )
    start_step = get_next_simulation_step(paths, artifacts.subject, user_id)
    new_attempts = build_manual_attempts(
        questions=artifacts.questions,
        evaluations=evaluations,
        start_step=start_step,
        user_id=user_id,
    )
    append_attempts(paths, artifacts.subject, user_id, new_attempts)

    combined = pd.concat([prior_attempts, new_attempts], ignore_index=True)
    after_snapshot = compute_live_mastery_snapshot(
        base_attempts=pd.DataFrame(),
        manual_attempts=combined,
        mastery_engine=services.mastery_engine,
        fallback_snapshot=fallback,
    )

    score = sum(1 for row in evaluations if row["correct"])
    st.success(f"Recorded {len(new_attempts)} attempts. Score: {score}/{len(evaluations)}. Progress saved.")

    touched_concepts = new_attempts["concept_id"].unique().tolist()
    before_scores = before_snapshot.set_index("concept_id")["graph_adjusted_mastery"].to_dict()
    after_rows = after_snapshot[after_snapshot["concept_id"].isin(touched_concepts)].copy()
    after_rows["delta"] = after_rows["concept_id"].map(
        lambda cid: round(
            float(after_rows.set_index("concept_id").loc[cid, "graph_adjusted_mastery"])
            - before_scores.get(cid, 0.0),
            4,
        )
    )
    st.subheader("Mastery Impact")
    mdf = after_rows[
        ["concept_name", "mastery_band", "graph_adjusted_mastery", "delta", "explanation"]
    ].copy()
    show_table(mdf, column_config=column_config_mastery(mdf), key="test_mastery_impact", height=280)
