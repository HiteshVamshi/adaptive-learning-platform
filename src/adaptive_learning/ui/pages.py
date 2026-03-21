"""Streamlit page renderers — module facade for `adaptive_learning.ui.pages`.

Implementation lives in `page_views/` so this path is always a single `.py` file.
That avoids environments where a stale `pages.py` shadowed a `pages/` package and
missed symbols such as `render_learn_page`.
"""

from __future__ import annotations

from adaptive_learning.ui.page_views import (
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

__all__ = [
    "render_analysis_dashboard",
    "render_content_generation_page",
    "render_curriculum_page",
    "render_debug_page",
    "render_learn_page",
    "render_practice_page",
    "render_search_page",
    "render_test_page",
    "render_tutor_page",
    "render_user_dashboard",
    "session_keys",
]
