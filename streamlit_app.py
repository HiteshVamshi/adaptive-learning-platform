from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.ui.data_access import (
    build_platform_services,
    compute_live_mastery_snapshot,
    default_paths,
    load_platform_artifacts,
)
from adaptive_learning.ui.pages import (
    render_analysis_dashboard,
    render_content_generation_page,
    render_curriculum_page,
    render_debug_page,
    render_practice_page,
    render_search_page,
    render_test_page,
    render_tutor_page,
)


@st.cache_resource
def get_artifacts():
    paths = default_paths(ROOT)
    return load_platform_artifacts(paths=paths)


@st.cache_resource
def get_services():
    paths = default_paths(ROOT)
    artifacts = get_artifacts()
    return build_platform_services(paths=paths, artifacts=artifacts)


def main() -> None:
    st.set_page_config(
        page_title="Adaptive Learning Platform",
        page_icon="book",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    artifacts = get_artifacts()
    services = get_services()

    if "manual_attempts" not in st.session_state:
        st.session_state["manual_attempts"] = pd.DataFrame()
    if "test_seed" not in st.session_state:
        st.session_state["test_seed"] = 13

    live_snapshot = compute_live_mastery_snapshot(
        base_attempts=artifacts.attempts,
        manual_attempts=st.session_state["manual_attempts"],
        mastery_engine=services.mastery_engine,
        fallback_snapshot=artifacts.mastery_snapshot,
    )

    st.sidebar.title("Adaptive Learning")
    st.sidebar.caption("Portfolio-grade local edtech system")
    page = st.sidebar.radio(
        "Navigate",
        options=[
            "Curriculum & Data",
            "Search",
            "Practice",
            "Test",
            "Analysis Dashboard",
            "AI Tutor",
            "Content Generation Demo",
            "System Debug View",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.metric("Concepts", int(len(live_snapshot)))
    st.sidebar.metric("Questions", int(len(artifacts.questions)))
    st.sidebar.metric("Theory Notes", int(len(artifacts.theory_content)))
    st.sidebar.metric("Manual Attempts This Session", int(len(st.session_state["manual_attempts"])))

    if page == "Curriculum & Data":
        render_curriculum_page(artifacts=artifacts)
    elif page == "Search":
        render_search_page(artifacts=artifacts, services=services)
    elif page == "Practice":
        render_practice_page(artifacts=artifacts, services=services, live_snapshot=live_snapshot)
    elif page == "Test":
        render_test_page(artifacts=artifacts, services=services)
    elif page == "Analysis Dashboard":
        render_analysis_dashboard(artifacts=artifacts, live_snapshot=live_snapshot)
    elif page == "AI Tutor":
        render_tutor_page(services=services)
    elif page == "Content Generation Demo":
        render_content_generation_page(artifacts=artifacts, services=services)
    elif page == "System Debug View":
        render_debug_page(artifacts=artifacts, services=services, live_snapshot=live_snapshot)


if __name__ == "__main__":
    main()
