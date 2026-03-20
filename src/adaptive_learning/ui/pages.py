from __future__ import annotations

import pandas as pd
import streamlit as st

from adaptive_learning.ui.data_access import (
    PlatformArtifacts,
    PlatformServices,
    build_chapter_coverage_table,
    build_manual_attempts,
    compute_live_mastery_snapshot,
    target_difficulty_for_band,
)
from adaptive_learning.ui.visualization import graph_to_dot, parse_pipe_list


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
    st.dataframe(sources_df, use_container_width=True, hide_index=True)

    coverage_df = build_chapter_coverage_table(artifacts=artifacts)
    st.subheader("Coverage by Official Chapter")
    st.dataframe(
        coverage_df[
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
        ],
        use_container_width=True,
        hide_index=True,
    )

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
        st.dataframe(
            syllabus_df[
                [
                    "unit_name",
                    "chapter_name",
                    "topic_name",
                    "unit_marks",
                    "periods",
                    "official_text",
                    "source_url",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

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
        st.dataframe(
            textbook_df[
                [
                    "chapter_number",
                    "chapter_name",
                    "section_number",
                    "section_title",
                    "start_page",
                    "chapter_pdf_url",
                    "source_url",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with theory_tab:
        theory_chapter = st.selectbox(
            "Filter theory notes by chapter",
            options=["All chapters"] + sorted(artifacts.theory_content["chapter_name"].dropna().unique().tolist()),
            key="theory_chapter",
        )
        theory_df = artifacts.theory_content.copy()
        if theory_chapter != "All chapters":
            theory_df = theory_df[theory_df["chapter_name"] == theory_chapter]
        st.dataframe(
            theory_df[
                [
                    "chapter_name",
                    "concept_name",
                    "title",
                    "content_type",
                    "textbook_sections",
                    "source_url",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Theory Note Preview"):
            preview_df = theory_df[["title", "summary_text", "official_syllabus_text"]].head(5).copy()
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

    with data_tab:
        source_counts = (
            artifacts.questions["source"].value_counts().rename_axis("source").reset_index(name="question_count")
        )
        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("Question Source Mix")
            st.dataframe(source_counts, use_container_width=True, hide_index=True)
        with right_col:
            st.subheader("Question Coverage by Chapter")
            st.dataframe(
                _question_source_summary(artifacts=artifacts),
                use_container_width=True,
                hide_index=True,
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

        st.dataframe(
            questions_df[
                [
                    "question_id",
                    "chapter_name",
                    "concept_name",
                    "difficulty",
                    "source",
                    "prompt",
                    "answer",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Concept Table"):
            concept_df = artifacts.concepts.copy()
            st.dataframe(
                concept_df[
                    [
                        "concept_id",
                        "name",
                        "node_type",
                        "chapter_name",
                        "source_kind",
                        "source_url",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

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
            st.dataframe(
                pd.DataFrame(
                    [
                        {"table_name": table_name, "row_count": row_count}
                        for table_name, row_count in db_status["table_counts"].items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.subheader("Stored Dataset Metadata")
            st.json(db_status["metadata"])


def render_search_page(*, artifacts: PlatformArtifacts, services: PlatformServices) -> None:
    st.title("Search")
    st.caption("Hybrid retrieval over CBSE Class 10 Mathematics questions and worked solutions.")

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
    st.dataframe(
        result_df[
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
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Retrieval Score Comparison")
    score_df = result_df.set_index("question_id")[["bm25_score", "vector_score", "hybrid_score"]]
    st.bar_chart(score_df)

    with st.expander("Answer Keys for Retrieved Results", expanded=False):
        st.dataframe(
            result_df[["question_id", "final_answer"]],
            use_container_width=True,
            hide_index=True,
        )


def render_practice_page(
    *,
    artifacts: PlatformArtifacts,
    services: PlatformServices,
    live_snapshot: pd.DataFrame,
) -> None:
    st.title("Practice")
    st.caption("Adaptive practice driven by mastery, the knowledge graph, and content generation.")

    weakest = live_snapshot.sort_values(by="graph_adjusted_mastery").head(5)
    st.subheader("Current Weak Concepts")
    st.dataframe(
        weakest[
            ["concept_name", "chapter_name", "graph_adjusted_mastery", "mastery_band", "explanation"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    concept_options = _concept_options(artifacts.concepts)
    default_concept = weakest.iloc[0]["concept_id"] if not weakest.empty else concept_options[0][1]
    selected_label = st.selectbox(
        "Practice concept",
        options=concept_options,
        index=_option_index(concept_options, default_concept),
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
    st.dataframe(
        recommendations[
            [
                "recommendation_rank",
                "question_id",
                "concept_name",
                "difficulty",
                "recommendation_score",
                "rationale",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Generate Fresh Practice", key="practice_generate"):
        generated_record, prompt_template = services.content_generator.generate_question(
            concept_id=concept_id,
            difficulty=difficulty,
        )
        st.session_state["practice_generated_question"] = {
            "prompt_template": prompt_template,
            "output": generated_record.to_dict(),
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


def render_test_page(
    *,
    artifacts: PlatformArtifacts,
    services: PlatformServices,
) -> None:
    st.title("Test")
    st.caption("Run a short quiz, record outcomes, and preview how mastery changes.")

    concept_options = _concept_options(artifacts.concepts)
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
        options=[("All concepts", "")] + concept_options,
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

    seed = st.session_state.get("test_seed", 13)
    sample_size = min(num_questions, len(question_pool))
    sampled_questions = question_pool.sample(
        n=sample_size,
        random_state=seed,
        replace=False,
    ).sort_values(by=["chapter_name", "concept_name", "difficulty", "question_id"])

    st.button("Shuffle Test Set", on_click=lambda: st.session_state.update({"test_seed": seed + 1}))

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

    prior_manual_attempts = st.session_state.get("manual_attempts", pd.DataFrame())
    before_snapshot = compute_live_mastery_snapshot(
        base_attempts=artifacts.attempts,
        manual_attempts=prior_manual_attempts,
        mastery_engine=services.mastery_engine,
        fallback_snapshot=artifacts.mastery_snapshot,
    )
    start_step = len(artifacts.attempts) + len(prior_manual_attempts)
    user_id = str(artifacts.mastery_snapshot["user_id"].iloc[0])
    new_attempts = build_manual_attempts(
        questions=artifacts.questions,
        evaluations=evaluations,
        start_step=start_step,
        user_id=user_id,
    )
    combined_manual_attempts = pd.concat([prior_manual_attempts, new_attempts], ignore_index=True)
    st.session_state["manual_attempts"] = combined_manual_attempts

    after_snapshot = compute_live_mastery_snapshot(
        base_attempts=artifacts.attempts,
        manual_attempts=combined_manual_attempts,
        mastery_engine=services.mastery_engine,
        fallback_snapshot=artifacts.mastery_snapshot,
    )

    score = sum(1 for row in evaluations if row["correct"])
    st.success(f"Recorded {len(new_attempts)} attempts. Score: {score}/{len(evaluations)}.")

    touched_concepts = new_attempts["concept_id"].unique().tolist()
    before_scores = before_snapshot.set_index("concept_id")["graph_adjusted_mastery"].to_dict()
    after_rows = after_snapshot[after_snapshot["concept_id"].isin(touched_concepts)].copy()
    after_rows["delta"] = after_rows["concept_id"].map(
        lambda concept_id: round(
            float(
                after_rows.set_index("concept_id").loc[concept_id, "graph_adjusted_mastery"]
                - before_scores.get(concept_id, 0.0)
            ),
            4,
        )
    )
    st.subheader("Mastery Impact")
    st.dataframe(
        after_rows[
            ["concept_name", "mastery_band", "graph_adjusted_mastery", "delta", "explanation"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_analysis_dashboard(
    *,
    artifacts: PlatformArtifacts,
    live_snapshot: pd.DataFrame,
) -> None:
    st.title("Analysis Dashboard")
    st.caption("Mastery, recommendations, and concept-graph structure for the current learner state.")

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
        st.dataframe(
            weakest[["concept_name", "chapter_name", "graph_adjusted_mastery", "mastery_band"]],
            use_container_width=True,
            hide_index=True,
        )
    with right_col:
        st.subheader("Strongest Concepts")
        strongest = live_snapshot.sort_values(by="graph_adjusted_mastery", ascending=False).head(7)
        st.dataframe(
            strongest[["concept_name", "chapter_name", "graph_adjusted_mastery", "mastery_band"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Mastery by Concept")
    chart_df = live_snapshot.set_index("concept_name")[["graph_adjusted_mastery"]]
    st.bar_chart(chart_df)

    concept_options = _concept_options(artifacts.concepts)
    selected_concept = st.selectbox(
        "Mastery history concept",
        options=concept_options,
        format_func=lambda item: item[0],
        key="history_concept",
    )
    history_df = artifacts.mastery_history[
        artifacts.mastery_history["concept_id"] == selected_concept[1]
    ].copy()
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
        options=[("Chapter overview", "")] + concept_options,
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


def render_tutor_page(*, services: PlatformServices) -> None:
    st.title("AI Tutor")
    st.caption("Chat with modular agents and inspect the exact tools they used.")

    agent_name = st.selectbox(
        "Agent mode",
        options=["tutor", "practice", "query"],
        format_func=lambda value: value.title(),
    )
    history_key = f"chat_history_{agent_name}"
    chat_history = st.session_state.setdefault(history_key, [])

    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("trace"):
                with st.expander("Tool Trace"):
                    st.json(message["trace"])

    prompt = st.chat_input(f"Ask the {agent_name.title()} Agent")
    if not prompt:
        return

    chat_history.append({"role": "user", "content": prompt})
    agent = services.agent_suite.get(agent_name)
    response = agent.respond(prompt)
    assistant_message = {
        "role": "assistant",
        "content": response.answer,
        "trace": [trace.to_dict() for trace in response.tool_trace],
        "metadata": response.metadata,
    }
    chat_history.append(assistant_message)

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response.answer)
        with st.expander("Tool Trace", expanded=True):
            st.json(assistant_message["trace"])
        with st.expander("Metadata"):
            st.json(response.metadata)


def render_content_generation_page(
    *,
    artifacts: PlatformArtifacts,
    services: PlatformServices,
) -> None:
    st.title("Content Generation Demo")
    st.caption("Grounded generation for questions, explanations, summaries, and difficulty adaptation.")

    task = st.selectbox("Generation task", options=["question", "explanation", "summary"])
    concept_options = _concept_options(artifacts.concepts)

    if task == "summary":
        scope_type = st.selectbox("Summary scope", options=["concept", "chapter"])
        if scope_type == "concept":
            selected = st.selectbox(
                "Concept",
                options=concept_options,
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
                    for row in artifacts.concepts[artifacts.concepts["node_type"] == "concept"].to_dict(orient="records")
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
        options=concept_options,
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
        st.subheader("Fine-Tuning / Adaptation View")
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
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.caption(comparison.adaptation_notes)

    with st.expander("Prompt Template"):
        st.code(prompt_template)


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
            st.dataframe(
                pd.DataFrame(
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
                ),
                use_container_width=True,
                hide_index=True,
            )

        with debug_cols[1]:
            st.subheader("RAG Context")
            st.markdown(rag_response.answer.answer)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "chunk_id": row.chunk_id,
                            "type": row.chunk_type,
                            "concept": row.concept_name,
                            "score": row.score,
                        }
                        for row in rag_response.retrieved_chunks
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

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
    st.dataframe(
        live_snapshot.sort_values(by="graph_adjusted_mastery").head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Adaptation Comparisons")
    st.dataframe(artifacts.adaptation_comparisons, use_container_width=True, hide_index=True)


def _question_source_summary(*, artifacts: PlatformArtifacts) -> pd.DataFrame:
    summary_df = (
        artifacts.questions.groupby(["chapter_name", "source"])["question_id"]
        .count()
        .reset_index(name="question_count")
        .sort_values(by=["chapter_name", "source"])
    )
    return summary_df


def _concept_options(concepts: pd.DataFrame) -> list[tuple[str, str]]:
    concept_rows = concepts[concepts["node_type"] == "concept"].copy()
    concept_rows = concept_rows.sort_values(by=["chapter_name", "order_index", "name"])
    return [
        (f"{row['chapter_name']} / {row['name']}", str(row["concept_id"]))
        for row in concept_rows.to_dict(orient="records")
    ]


def _option_index(options: list[tuple[str, str]], target_value: str) -> int:
    for index, (_, value) in enumerate(options):
        if value == target_value:
            return index
    return 0
