from __future__ import annotations

import streamlit as st

from adaptive_learning.ui.data_access import PlatformServices

__all__ = ["render_tutor_page"]


def render_tutor_page(
    *,
    services: PlatformServices,
    subject_label: str,
    live_mastery_snapshot=None,
) -> None:
    st.title("AI Tutor")
    st.caption(f"Chat with modular agents for {subject_label} and inspect the exact tools they used.")

    agent_name = st.selectbox(
        "Agent mode",
        options=["tutor", "practice", "query", "learn"],
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
    kwargs = {}
    if agent_name in ("tutor", "learn") and live_mastery_snapshot is not None:
        kwargs["live_mastery_snapshot"] = live_mastery_snapshot
    response = agent.respond(prompt, **kwargs)
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
            st.json(assistant_message["metadata"])
