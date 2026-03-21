"""Streamlit page renderers split by feature area."""

from __future__ import annotations

from adaptive_learning.ui.page_views.analysis import render_analysis_dashboard
from adaptive_learning.ui.page_views.content_generation import render_content_generation_page
from adaptive_learning.ui.page_views.curriculum import render_curriculum_page
from adaptive_learning.ui.page_views.learn_page import render_learn_page
from adaptive_learning.ui.page_views.debug import render_debug_page
from adaptive_learning.ui.page_views.practice import render_practice_page
from adaptive_learning.ui.page_views.search import render_search_page
from adaptive_learning.ui.page_views.session import session_keys
from adaptive_learning.ui.page_views.test_page import render_test_page
from adaptive_learning.ui.page_views.tutor import render_tutor_page
from adaptive_learning.ui.page_views.user_dashboard import render_user_dashboard

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
