from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.config import ui_mode
from adaptive_learning.ui.table_display import inject_app_table_styles
from adaptive_learning.deployment.runtime import ensure_platform_ready
from adaptive_learning.ui.data_access import (
    build_empty_mastery_snapshot,
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
    render_learn_page,
    render_practice_page,
    render_search_page,
    render_test_page,
    render_tutor_page,
    render_user_dashboard,
    session_keys,
)
from adaptive_learning.ui.user_progress import (
    get_or_create_user,
    get_user_attempts,
    list_users,
)

SUBJECT_OPTIONS = {
    "math": "Mathematics",
    "science": "Science",
}

# (page_id, sidebar label) — order is the suggested learning flow
_NAV_LEARNER: list[tuple[str, str]] = [
    ("progress", "📊  My progress"),
    ("learn", "📖  Learn"),
    ("practice", "✏️  Practice"),
    ("test", "📝  Test yourself"),
    ("curriculum", "📚  Syllabus & textbook"),
    ("search", "🔍  Search questions"),
    ("tutor", "💬  AI tutor"),
]

_NAV_FULL_EXTRA: list[tuple[str, str]] = [
    ("analysis", "📈  Analytics dashboard"),
    ("content", "🧪  Content generation demo"),
    ("debug", "🔧  System debug"),
]


@st.cache_resource
def get_deployment_status(subject: str):
    paths = default_paths(ROOT, subject=subject)
    return ensure_platform_ready(paths=paths)


@st.cache_resource
def get_artifacts(subject: str):
    paths = default_paths(ROOT, subject=subject)
    get_deployment_status(subject)
    return load_platform_artifacts(paths=paths)


@st.cache_resource
def get_services(subject: str):
    paths = default_paths(ROOT, subject=subject)
    artifacts = get_artifacts(subject)
    return build_platform_services(paths=paths, artifacts=artifacts)


def _nav_choices(mode: str) -> list[tuple[str, str]]:
    if mode == "full":
        return _NAV_LEARNER + _NAV_FULL_EXTRA
    return list(_NAV_LEARNER)


def main() -> None:
    mode = ui_mode()
    st.set_page_config(
        page_title="Class 10 Learning",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_app_table_styles()

    st.sidebar.title("Class 10 study")
    if mode == "learner":
        st.sidebar.caption("Pick your subject, sign in, then choose where to go.")
    else:
        st.sidebar.caption("Learner UI + educator / developer tools (`ADAPTIVE_LEARNING_UI_MODE=full`).")

    subject = st.sidebar.selectbox(
        "Subject",
        options=list(SUBJECT_OPTIONS.keys()),
        format_func=lambda value: SUBJECT_OPTIONS[value],
    )

    deployment_status = get_deployment_status(subject)
    if not deployment_status.artifacts_ready:
        st.error("Course materials are not ready yet (build step did not finish or auto-build is off).")
        if deployment_status.missing_paths:
            st.code("\n".join(deployment_status.missing_paths))
        if mode == "full":
            st.info("From the project root run `python scripts/build_all.py` or enable auto-build in deployment settings.")
        return

    paths = default_paths(ROOT, subject=subject)
    artifacts = get_artifacts(subject)
    services = get_services(subject)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sign in")
    existing = list_users(paths, subject)
    opts = ["New learner…"] + [f"{name}" for _, name in existing]
    uids = [None] + [uid for uid, _ in existing]

    pick_label = st.sidebar.selectbox("Returning learner", options=opts, key="user_pick")
    new_name = st.sidebar.text_input(
        "Or create a profile (your name)",
        placeholder="e.g. Alex",
        key="user_new_name",
    )
    if st.sidebar.button("Continue", type="primary"):
        idx = opts.index(pick_label) if pick_label in opts else 0
        if idx > 0 and uids[idx]:
            st.session_state["current_user_id"] = uids[idx]
            st.session_state["current_user_name"] = pick_label
        elif new_name and str(new_name).strip():
            uid = get_or_create_user(paths, subject, str(new_name).strip())
            st.session_state["current_user_id"] = uid
            st.session_state["current_user_name"] = str(new_name).strip()
        else:
            st.sidebar.warning("Choose an existing learner or enter a name.")

    user_id = st.session_state.get("current_user_id")
    user_name = st.session_state.get("current_user_name", "Learner")

    if not user_id:
        st.info("**Welcome.** In the sidebar, choose **New learner** or your name, then click **Continue**.")
        return

    user_attempts = get_user_attempts(paths, subject, user_id)
    base_attempts = pd.DataFrame()
    fallback = build_empty_mastery_snapshot(artifacts.concepts, artifacts.mastery_snapshot)
    live_snapshot = compute_live_mastery_snapshot(
        base_attempts=base_attempts,
        manual_attempts=user_attempts,
        mastery_engine=services.mastery_engine,
        fallback_snapshot=fallback,
    )
    user_mastery_history: pd.DataFrame | None = None
    if not user_attempts.empty:
        run_result = services.mastery_engine.run(attempts=user_attempts)
        user_mastery_history = run_result.history

    test_seed_key = f"test_seed_{subject}"
    if test_seed_key not in st.session_state:
        st.session_state[test_seed_key] = 13

    nav = _nav_choices(mode)
    labels = [label for _, label in nav]
    if st.session_state.get("nav_page_label") not in labels:
        st.session_state["nav_page_label"] = labels[0]
    chosen = st.sidebar.selectbox(
        "Go to",
        options=labels,
        key="nav_page_label",
    )
    page_id = next(pid for pid, lab in nav if lab == chosen)

    st.sidebar.markdown("---")
    st.sidebar.metric("Signed in as", user_name)
    st.sidebar.metric("Subject", SUBJECT_OPTIONS[subject])
    st.sidebar.metric("Topics tracked", int(len(live_snapshot)))
    st.sidebar.metric("Your attempts", int(len(user_attempts)))

    if mode == "full":
        with st.sidebar.expander("Developer / troubleshooting", expanded=False):
            st.caption(
                "Use after running `build_all`, `generate_data`, or `build_learn_index` while the app stayed open."
            )
            if st.button("Clear app cache & reload", key="dev_clear_streamlit_cache"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
            st.caption(f"Artifacts: `{paths.artifacts_dir}`")
            st.caption(
                f"Embeddings: {deployment_status.embedding_backend} · "
                f"Auto-built: {'yes' if deployment_status.auto_built else 'no'}"
            )

    st.title(f"{SUBJECT_OPTIONS[subject]} · CBSE Class 10")
    st.caption(f"Hello, **{user_name}**. Use the sidebar to move between study areas.")

    if page_id == "progress":
        render_user_dashboard(
            artifacts=artifacts,
            live_snapshot=live_snapshot,
            user_display_name=user_name,
            total_attempts=int(len(user_attempts)),
        )
    elif page_id == "learn":
        render_learn_page(
            artifacts=artifacts,
            live_snapshot=live_snapshot,
            user_attempts=user_attempts,
            paths=paths,
        )
    elif page_id == "curriculum":
        render_curriculum_page(artifacts=artifacts)
    elif page_id == "search":
        render_search_page(artifacts=artifacts, services=services)
    elif page_id == "practice":
        session_keys.manual_attempts_key = None
        session_keys.test_seed_key = test_seed_key
        render_practice_page(
            artifacts=artifacts,
            services=services,
            live_snapshot=live_snapshot,
            user_id=user_id,
            paths=paths,
        )
    elif page_id == "test":
        session_keys.manual_attempts_key = None
        session_keys.test_seed_key = test_seed_key
        render_test_page(
            artifacts=artifacts,
            services=services,
            user_id=user_id,
            paths=paths,
        )
    elif page_id == "analysis":
        render_analysis_dashboard(
            artifacts=artifacts,
            live_snapshot=live_snapshot,
            user_mastery_history=user_mastery_history,
        )
    elif page_id == "tutor":
        render_tutor_page(
            services=services,
            subject_label=SUBJECT_OPTIONS[subject],
            live_mastery_snapshot=live_snapshot,
        )
    elif page_id == "content":
        render_content_generation_page(artifacts=artifacts, services=services)
    elif page_id == "debug":
        render_debug_page(artifacts=artifacts, services=services, live_snapshot=live_snapshot)


if __name__ == "__main__":
    main()
