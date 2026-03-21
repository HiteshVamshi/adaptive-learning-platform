from __future__ import annotations

from pathlib import Path

from adaptive_learning.learn.io import write_learn_resources_csv
from adaptive_learning.learn.sources import seed_for_subject


def write_learn_index_for_paths(paths) -> Path:
    rows = [r.to_dict() for r in seed_for_subject(paths.subject)]
    return write_learn_resources_csv(rows, paths.artifacts_dir)


def write_learn_index_for_subject(*, artifacts_dir: Path, subject: str) -> Path:
    rows = [r.to_dict() for r in seed_for_subject(subject)]
    return write_learn_resources_csv(rows, artifacts_dir)
