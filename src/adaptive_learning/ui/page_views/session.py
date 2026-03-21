from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionKeys:
    """Mutable keys so Streamlit can scope session state per subject."""

    manual_attempts_key: str = "manual_attempts"
    test_seed_key: str = "test_seed"


session_keys = SessionKeys()
