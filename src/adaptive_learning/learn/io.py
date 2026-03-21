from __future__ import annotations

from pathlib import Path

import pandas as pd


def learn_resources_csv_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "learn_resources.csv"


def load_learn_resources(artifacts_dir: Path) -> pd.DataFrame:
    path = learn_resources_csv_path(artifacts_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_learn_resources_csv(rows: list[dict], artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = learn_resources_csv_path(artifacts_dir)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
