from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentConfig:
    embedding_backend: str = "hash"
    auto_build: bool = True
    sqlite_filename: str = "adaptive_learning.db"

    @classmethod
    def from_env(cls) -> DeploymentConfig:
        auto_build_value = os.getenv("ADAPTIVE_LEARNING_AUTO_BUILD", "true").strip().lower()
        auto_build = auto_build_value not in {"0", "false", "no"}
        embedding_backend = os.getenv("ADAPTIVE_LEARNING_EMBEDDING_BACKEND", "hash").strip() or "hash"
        sqlite_filename = (
            os.getenv("ADAPTIVE_LEARNING_SQLITE_FILENAME", "adaptive_learning.db").strip() or "adaptive_learning.db"
        )
        return cls(
            embedding_backend=embedding_backend,
            auto_build=auto_build,
            sqlite_filename=sqlite_filename,
        )


@dataclass(frozen=True)
class DeploymentStatus:
    artifacts_ready: bool
    auto_built: bool
    missing_paths: list[str]
    embedding_backend: str
