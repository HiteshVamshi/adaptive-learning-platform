from __future__ import annotations

import pandas as pd
import streamlit as st

from adaptive_learning.ui.data_access import (
    PlatformArtifacts,
    PlatformServices,
    build_practice_attempt,
    target_difficulty_for_band,
)
from adaptive_learning.ui.pages._common import concept_options, option_index
from adaptive_learning.ui.table_display import column_config_mastery, column_config_recommendations, show_table
from adaptive_learning.ui.user_progress import append_attempts, get_next_simulation_step

__all__ = ["render_practice_page"]


def render_practice_page(
    *,
    artifacts: PlatformArtifacts,
    services: PlatformServices,
    live_snapshot: pd.DataFrame,
    user_id: str,
    paths,
) -> None:
    st.title("Practice")
    st.caption("Adaptive practice: attempt questions and record outcomes. Mastery updates live.")

    weakest = live_snapshot.sort_values(by="graph_adjusted_mastery").head(5)
    st.subheader("Current Weak Concepts")
    wdf = weakest[
        ["concept_name", "chapter_name", "graph_adjusted_mastery", "mastery_band", "explanation"]
    ].copy()
    show_table(wdf, column_config=column_config_mastery(wdf), key="practice_weak", height=260)

    opts = concept_options(artifacts.concepts)
    default_concept = weakest.iloc[0]["concept_id"] if not weakest.empty else opts[0][1]
    selected_label = st.selectbox(
        "Practice concept",
        options=opts,
        index=option_index(opts, default_concept),
        format_func=lambda item: item[0],
    )
    concept_id = selected_label[1]

    concept_mastery = live_snapshot[live_snapshot["concept_id"] == concept_id]
    mastery_band = (
        str(concept_mastery.iloc[0]["mastery_band"])
        if not concept_mastery.empty
        else "developing"
    )
    recommended_difficulty = target_difficulty_for_band(mastery_band)
    difficulty = st.selectbox(
        "Generated difficulty",
        options=["easy", "medium", "hard"],
        index=["easy", "medium", "hard"].index(recommended_difficulty),
    )

    recommendations = artifacts.recommendations[artifacts.recommendations["concept_id"] == concept_id].head(5)
    if recommendations.empty:
        recommendations = artifacts.recommendations.head(5)

    st.subheader("Recommended Existing Questions")
    rdf = recommendations[
        [
            "recommendation_rank",
            "question_id",
            "concept_name",
            "difficulty",
            "recommendation_score",
            "rationale",
        ]
    ].copy()
    show_table(rdf, column_config=column_config_recommendations(rdf), key="practice_recs", height=320)

    if st.button("Generate Fresh Practice", key="practice_generate"):
        generated_record, prompt_template = services.content_generator.generate_question(
            concept_id=concept_id,
            difficulty=difficulty,
        )
        st.session_state["practice_generated_question"] = {
            "prompt_template": prompt_template,
            "output": generated_record.to_dict(),
            "concept_id": concept_id,
            "concept_name": generated_record.concept_name,
            "chapter_name": generated_record.chapter_name,
            "difficulty": difficulty,
        }

    generated_payload = st.session_state.get("practice_generated_question")
    if generated_payload:
        output = generated_payload["output"]
        st.subheader("Generated Practice")
        st.markdown(f"**Prompt**  \n{output['prompt']}")
        st.markdown(f"**Final answer**  \n{output['final_answer']}")
        st.markdown(f"**Explanation**  \n{output['explanation']}")
        st.caption(f"Tags: {output['tags']}")
        with st.expander("Prompt Template"):
            st.code(generated_payload["prompt_template"])

        st.markdown("---")
        st.subheader("Record your attempt")
        correct = st.radio("Did you get it right?", options=["Correct", "Incorrect"], horizontal=True)
        response_time = st.number_input("Response time (seconds)", min_value=20, max_value=1200, value=90, step=10)
        if st.button("Save attempt", key="practice_save"):
            step = get_next_simulation_step(paths, artifacts.subject, user_id)
            attempt_df = build_practice_attempt(
                concept_id=generated_payload["concept_id"],
                concept_name=generated_payload["concept_name"],
                chapter_name=generated_payload["chapter_name"],
                difficulty=generated_payload["difficulty"],
                correct=(correct == "Correct"),
                response_time_sec=response_time,
                user_id=user_id,
                simulation_step=step,
            )
            append_attempts(paths, artifacts.subject, user_id, attempt_df)
            st.success("Attempt recorded. Check **My Progress** to see mastery change.")
            del st.session_state["practice_generated_question"]
            st.rerun()
