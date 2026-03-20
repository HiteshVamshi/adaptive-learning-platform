from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_learning.deployment.runtime import DeploymentConfig, ensure_platform_ready
from adaptive_learning.ui.data_access import default_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all local artifacts needed by the adaptive learning platform."
    )
    parser.add_argument(
        "--embedding-backend",
        default="hash",
        choices=["hash", "auto", "sentence_transformer"],
        help="Embedding backend used for search and RAG indices.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all artifacts even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_paths(ROOT)
    config = DeploymentConfig(
        embedding_backend=args.embedding_backend,
        auto_build=True,
    )
    status = ensure_platform_ready(paths=paths, config=config, force_rebuild=args.force)
    print(
        f"artifacts_ready={status.artifacts_ready} "
        f"auto_built={status.auto_built} "
        f"embedding_backend={status.embedding_backend}"
    )
    if status.missing_paths:
        print("missing_paths=")
        for path in status.missing_paths:
            print(path)


if __name__ == "__main__":
    main()
