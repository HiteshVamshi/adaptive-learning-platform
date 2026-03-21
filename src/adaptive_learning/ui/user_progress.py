"""
Persistent user progress store: identities and attempt records per subject.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from adaptive_learning.mastery.schemas import AttemptRecord

_ATTEMPTS_COLUMNS = [
    "attempt_id",
    "user_id",
    "simulation_step",
    "timestamp",
    "question_id",
    "concept_id",
    "concept_name",
    "chapter_name",
    "difficulty",
    "correct",
    "response_time_sec",
    "expected_time_sec",
    "source",
]


def _user_progress_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "user_progress.db"


def ensure_store(paths) -> None:
    path = _user_progress_path(paths.artifacts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            create table if not exists users (
                user_id text primary key,
                display_name text not null,
                created_at text not null,
                subject text not null
            )
            """
        )
        conn.execute(
            """
            create table if not exists attempts (
                attempt_id text primary key,
                user_id text not null,
                subject text not null,
                simulation_step integer not null,
                timestamp text not null,
                question_id text not null,
                concept_id text not null,
                concept_name text not null,
                chapter_name text not null,
                difficulty text not null,
                correct integer not null,
                response_time_sec integer not null,
                expected_time_sec integer not null,
                source text not null,
                foreign key (user_id) references users(user_id)
            )
            """
        )
        conn.execute(
            "create index if not exists idx_attempts_user_subject on attempts(user_id, subject)"
        )
        conn.execute(
            "create index if not exists idx_attempts_user_subject_step on attempts(user_id, subject, simulation_step)"
        )
        conn.commit()


def list_users(paths, subject: str) -> list[tuple[str, str]]:
    """Return [(user_id, display_name), ...] for subject."""
    path = _user_progress_path(paths.artifacts_dir)
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "select user_id, display_name from users where subject = ? order by created_at",
            (subject,),
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_or_create_user(paths, subject: str, display_name: str) -> str:
    """Return user_id for display_name+subject; create if not exists."""
    ensure_store(paths)
    name = (display_name or "Learner").strip()
    path = _user_progress_path(paths.artifacts_dir)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "select user_id from users where display_name = ? and subject = ?",
            (name, subject),
        ).fetchone()
        if row:
            return row[0]
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name[:40])
        user_id = f"u_{subject}_{safe}_{uuid.uuid4().hex[:6]}"
        conn.execute(
            "insert into users (user_id, display_name, created_at, subject) values (?, ?, ?, ?)",
            (user_id, name, pd.Timestamp.utcnow().isoformat(), subject),
        )
        conn.commit()
    return user_id


def get_user_attempts(paths, subject: str, user_id: str) -> pd.DataFrame:
    """Load all attempts for user+subject as DataFrame matching AttemptRecord schema."""
    path = _user_progress_path(paths.artifacts_dir)
    if not path.exists():
        return pd.DataFrame(columns=_ATTEMPTS_COLUMNS)
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(
            """
            select attempt_id, user_id, simulation_step, timestamp, question_id,
                   concept_id, concept_name, chapter_name, difficulty, correct,
                   response_time_sec, expected_time_sec, source
            from attempts
            where user_id = ? and subject = ?
            order by simulation_step, timestamp
            """,
            conn,
            params=(user_id, subject),
        )
    if df.empty:
        return pd.DataFrame(columns=_ATTEMPTS_COLUMNS)
    return df


def append_attempts(paths, subject: str, user_id: str, attempts_df: pd.DataFrame) -> None:
    if attempts_df.empty:
        return
    ensure_store(paths)
    path = _user_progress_path(paths.artifacts_dir)
    rows = []
    for _, row in attempts_df.iterrows():
        rows.append((
            str(row["attempt_id"]),
            str(user_id),
            subject,
            int(row["simulation_step"]),
            str(row["timestamp"]),
            str(row["question_id"]),
            str(row["concept_id"]),
            str(row["concept_name"]),
            str(row["chapter_name"]),
            str(row["difficulty"]),
            int(row["correct"]),
            int(row["response_time_sec"]),
            int(row["expected_time_sec"]),
            str(row.get("source", "streamlit")),
        ))
    with sqlite3.connect(path) as conn:
        conn.executemany(
            """
            insert or replace into attempts
            (attempt_id, user_id, subject, simulation_step, timestamp, question_id,
             concept_id, concept_name, chapter_name, difficulty, correct,
             response_time_sec, expected_time_sec, source)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def get_next_simulation_step(paths, subject: str, user_id: str) -> int:
    df = get_user_attempts(paths, subject, user_id)
    if df.empty:
        return 1
    return int(df["simulation_step"].max()) + 1
