from __future__ import annotations

import pandas as pd
import streamlit as st

from adaptive_learning.ui.data_access import PlatformArtifacts, PlatformServices
from adaptive_learning.ui.pages._common import concept_options
from adaptive_learning.ui.table_display import column_config_adaptation_comparison, show_table
from adaptive_learning.ui.visualization import parse_pipe_list

__all__ = ["render_content_generation_page"]


def render_content_generation_page(
    *,
    artifacts: PlatformArtifacts,
    services: PlatformServices,
) -> None:
    st.title("Content Generation Demo")
    st.caption("Grounded generation for questions, explanations, summaries, and difficulty adaptation.")

    task = st.selectbox("Generation task", options=["question", "explanation", "summary"])
    opts = concept_options(artifacts.concepts)

    if task == "summary":
        scope_type = st.selectbox("Summary scope", options=["concept", "chapter"])
        if scope_type == "concept":
            selected = st.selectbox(
                "Concept",
                options=opts,
                format_func=lambda item: item[0],
            )
            record, prompt_template = services.content_generator.generate_summary(
                scope_type="concept",
                scope_id=selected[1],
            )
        else:
            chapter_options = sorted(
                {
                    (str(row["chapter_name"]), str(row["chapter_id"]))
                    for row in artifacts.concepts[artifacts.concepts["node_type"] == "concept"].to_dict(
                        orient="records"
                    )
                }
            )
            selected = st.selectbox(
                "Chapter",
                options=chapter_options,
                format_func=lambda item: item[0],
            )
            record, prompt_template = services.content_generator.generate_summary(
                scope_type="chapter",
                scope_id=selected[1],
            )
        st.markdown(f"**Title**  \n{record.title}")
        st.markdown(f"**Summary**  \n{record.summary}")
        st.write(parse_pipe_list("|".join(record.bullet_points)))
        with st.expander("Prompt Template"):
            st.code(prompt_template)
        return

    selected = st.selectbox(
        "Concept",
        options=opts,
        format_func=lambda item: item[0],
    )

    if task == "explanation":
        record, prompt_template = services.content_generator.generate_explanation(concept_id=selected[1])
        st.markdown(f"**Explanation**  \n{record.explanation}")
        st.write(record.key_points)
        with st.expander("Prompt Template"):
            st.code(prompt_template)
        return

    difficulty = st.selectbox("Target difficulty", options=["easy", "medium", "hard"], index=1)
    generated_record, prompt_template = services.content_generator.generate_question(
        concept_id=selected[1],
        difficulty=difficulty,
    )
    st.markdown(f"**Generated prompt**  \n{generated_record.prompt}")
    st.markdown(f"**Final answer**  \n{generated_record.final_answer}")
    st.markdown(f"**Explanation**  \n{generated_record.explanation}")
    st.caption(f"Tags: {' | '.join(generated_record.tags)}")

    if services.tuned_generator is not None:
        comparison = services.tuned_generator.compare_generation(
            concept_id=selected[1],
            target_difficulty=difficulty,
        )
        st.subheader("Difficulty adaptation (calibrator demo)")
        comparison_df = pd.DataFrame(
            [
                {
                    "variant": "baseline",
                    "predicted_label": comparison.baseline_prediction,
                    "target_probability": comparison.baseline_target_probability,
                    "prompt": comparison.baseline_prompt,
                },
                {
                    "variant": "adapted",
                    "predicted_label": comparison.adapted_prediction,
                    "target_probability": comparison.adapted_target_probability,
                    "prompt": comparison.adapted_prompt,
                },
            ]
        )
        show_table(
            comparison_df,
            column_config=column_config_adaptation_comparison(comparison_df),
            key="content_adapt_compare",
            height=220,
        )
        st.caption(comparison.adaptation_notes)

    with st.expander("Prompt Template"):
        st.code(prompt_template)
