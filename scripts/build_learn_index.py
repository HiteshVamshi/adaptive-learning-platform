"""Emit learn_resources.csv for one or all subjects (curated videos / NCERT PDFs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.data.generator import SUPPORTED_SUBJECTS
from adaptive_learning.learn.build_index import write_learn_index_for_subject
from adaptive_learning.ui.data_access import default_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learn_resources.csv per subject.")
    parser.add_argument(
        "--subject",
        default="all",
        choices=[*SUPPORTED_SUBJECTS, "all"],
        help="Subject to build (default: all).",
    )
    args = parser.parse_args()
    subjects = list(SUPPORTED_SUBJECTS) if args.subject == "all" else [args.subject]
    for subj in subjects:
        paths = default_paths(ROOT, subject=subj)
        out = write_learn_index_for_subject(artifacts_dir=paths.artifacts_dir, subject=subj)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
