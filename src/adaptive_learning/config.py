"""Central environment-driven preferences for deployment and UI."""

from __future__ import annotations

import os


def rag_generator_backend() -> str:
    """RAG answer generator: grounded | auto | transformers (see rag.generator)."""
    value = os.getenv("ADAPTIVE_LEARNING_RAG_GENERATOR_BACKEND", "").strip()
    return value or "grounded"


def content_generation_backend() -> str:
    value = os.getenv("ADAPTIVE_LEARNING_CONTENT_GENERATION_BACKEND", "").strip()
    return value or "grounded"


def ui_mode() -> str:
    """Streamlit shell: 'learner' (default) hides demos/debug/dev tools; 'full' shows everything."""
    value = os.getenv("ADAPTIVE_LEARNING_UI_MODE", "").strip().lower()
    if value in ("full", "developer", "dev"):
        return "full"
    return "learner"


def youtube_api_key() -> str | None:
    """YouTube Data API v3 key for programmatic learn-resource video search at index build time."""
    for key in ("YOUTUBE_API_KEY", "ADAPTIVE_LEARNING_YOUTUBE_API_KEY"):
        v = os.getenv(key, "").strip()
        if v:
            return v
    return None
