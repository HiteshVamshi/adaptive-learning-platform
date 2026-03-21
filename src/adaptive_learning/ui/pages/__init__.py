"""Streamlit page renderers split by feature area."""

from __future__ import annotations

from adaptive_learning.ui.pages.analysis import render_analysis_dashboard
from adaptive_learning.ui.pages.content_generation import render_content_generation_page
from adaptive_learning.ui.pages.curriculum import render_curriculum_page
from adaptive_learning.ui.pages.learn_page import render_learn_page
from adaptive_learning.ui.pages.debug import render_debug_page
from adaptive_learning.ui.pages.practice import render_practice_page
from adaptive_learning.ui.pages.search import render_search_page
from adaptive_learning.ui.pages.session import session_keys
from adaptive_learning.ui.pages.test_page import render_test_page
from adaptive_learning.ui.pages.tutor import render_tutor_page
from adaptive_learning.ui.pages.user_dashboard import render_user_dashboard

__all__ = [
    "render_analysis_dashboard",
    "render_user_dashboard",
    "render_content_generation_page",
    "render_curriculum_page",
    "render_learn_page",
    "render_debug_page",
    "render_practice_page",
    "render_search_page",
    "render_test_page",
    "render_tutor_page",
    "session_keys",
]
