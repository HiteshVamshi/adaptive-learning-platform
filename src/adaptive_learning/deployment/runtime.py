from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from adaptive_learning.content_generation.pipeline import build_content_bundle
from adaptive_learning.data.generator import generate_cbse_math_dataset
from adaptive_learning.data.io import write_dataset
from adaptive_learning.data.sqlite_store import write_dataset_to_sqlite
from adaptive_learning.fine_tuning.pipeline import run_fine_tuning_pipeline
from adaptive_learning.mastery.pipeline import run_mastery_pipeline
from adaptive_learning.rag.build_index import build_rag_index
from adaptive_learning.recommendation.pipeline import run_recommendation_pipeline
from adaptive_learning.search.build_index import build_search_index


@dataclass(frozen=True)
class DeploymentConfig:
    embedding_backend: str = "hash"
    auto_build: bool = True
    sqlite_filename: str = "adaptive_learning.db"

    @classmethod
    def from_env(cls) -> "DeploymentConfig":
        auto_build_value = os.getenv("ADAPTIVE_LEARNING_AUTO_BUILD", "true").strip().lower()
        auto_build = auto_build_value not in {"0", "false", "no"}
        embedding_backend = os.getenv("ADAPTIVE_LEARNING_EMBEDDING_BACKEND", "hash").strip() or "hash"
        sqlite_filename = os.getenv("ADAPTIVE_LEARNING_SQLITE_FILENAME", "adaptive_learning.db").strip() or "adaptive_learning.db"
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


def ensure_platform_ready(*, paths, config: DeploymentConfig | None = None, force_rebuild: bool = False) -> DeploymentStatus:
    config = config or DeploymentConfig.from_env()
    missing = missing_artifacts(paths=paths)
    if not force_rebuild and not missing:
        return DeploymentStatus(
            artifacts_ready=True,
            auto_built=False,
            missing_paths=[],
            embedding_backend=config.embedding_backend,
        )

    if not config.auto_build:
        return DeploymentStatus(
            artifacts_ready=False,
            auto_built=False,
            missing_paths=[str(path) for path in missing],
            embedding_backend=config.embedding_backend,
        )

    _build_all(paths=paths, config=config)
    still_missing = missing_artifacts(paths=paths)
    return DeploymentStatus(
        artifacts_ready=not still_missing,
        auto_built=True,
        missing_paths=[str(path) for path in still_missing],
        embedding_backend=config.embedding_backend,
    )


def missing_artifacts(*, paths) -> list[Path]:
    return [path for path in required_artifact_paths(paths=paths) if not path.exists()]


def required_artifact_paths(*, paths) -> list[Path]:
    return [
        paths.data_dir / "concepts.csv",
        paths.data_dir / "questions.csv",
        paths.data_dir / "solutions.csv",
        paths.data_dir / "concept_graph.json",
        paths.data_dir / "syllabus_topics.csv",
        paths.data_dir / "textbook_sections.csv",
        paths.data_dir / "theory_content.csv",
        paths.data_dir / "dataset_summary.json",
        paths.search_index_dir / "documents.csv",
        paths.search_index_dir / "faiss.index",
        paths.search_index_dir / "index_summary.json",
        paths.rag_index_dir / "documents.csv",
        paths.rag_index_dir / "faiss.index",
        paths.rag_index_dir / "rag_summary.json",
        paths.mastery_dir / "attempts.csv",
        paths.mastery_dir / "mastery_snapshot.csv",
        paths.mastery_dir / "mastery_history.csv",
        paths.mastery_dir / "mastery_summary.json",
        paths.recommendation_dir / "recommendations.csv",
        paths.recommendation_dir / "recommendation_summary.json",
        paths.generated_content_dir / "generated_questions.csv",
        paths.generated_content_dir / "generated_explanations.csv",
        paths.generated_content_dir / "generated_summaries.csv",
        paths.generated_content_dir / "content_generation_summary.json",
        paths.fine_tuning_dir / "adaptation_comparisons.csv",
        paths.fine_tuning_dir / "difficulty_calibrator.joblib",
        paths.fine_tuning_dir / "difficulty_metrics.json",
        paths.fine_tuning_dir / "fine_tuning_summary.json",
    ]


def _build_all(*, paths, config: DeploymentConfig) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.search_index_dir.mkdir(parents=True, exist_ok=True)
    paths.rag_index_dir.mkdir(parents=True, exist_ok=True)
    paths.mastery_dir.mkdir(parents=True, exist_ok=True)
    paths.recommendation_dir.mkdir(parents=True, exist_ok=True)
    paths.generated_content_dir.mkdir(parents=True, exist_ok=True)
    paths.fine_tuning_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_cbse_math_dataset()
    write_dataset(dataset=dataset, output_dir=paths.data_dir)
    write_dataset_to_sqlite(
        dataset=dataset,
        sqlite_path=paths.data_dir / config.sqlite_filename,
    )

    build_search_index(
        data_dir=paths.data_dir,
        output_dir=paths.search_index_dir,
        embedding_backend=config.embedding_backend,
    )
    build_rag_index(
        data_dir=paths.data_dir,
        output_dir=paths.rag_index_dir,
        embedding_backend=config.embedding_backend,
    )
    run_mastery_pipeline(
        data_dir=paths.data_dir,
        output_dir=paths.mastery_dir,
    )
    run_recommendation_pipeline(
        data_dir=paths.data_dir,
        mastery_dir=paths.mastery_dir,
        output_dir=paths.recommendation_dir,
        top_k=10,
    )
    build_content_bundle(
        data_dir=paths.data_dir,
        output_dir=paths.generated_content_dir,
        concept_ids=_default_generation_concepts(dataset.concepts),
        backend="grounded",
    )
    run_fine_tuning_pipeline(
        data_dir=paths.data_dir,
        output_dir=paths.fine_tuning_dir,
        concept_pairs=_default_fine_tuning_pairs(),
    )


def _default_generation_concepts(concepts_df) -> list[str]:
    concept_rows = concepts_df[concepts_df["node_type"] == "concept"].copy()
    preferred_ids = [
        "c_hcf_lcm",
        "c_trigonometric_ratios",
        "c_classical_probability",
    ]
    available = concept_rows["concept_id"].astype(str).tolist()
    selected = [concept_id for concept_id in preferred_ids if concept_id in available]
    if len(selected) == 3:
        return selected
    fallback = concept_rows.sort_values(by=["chapter_name", "order_index"]).head(3)
    return fallback["concept_id"].astype(str).tolist()


def _default_fine_tuning_pairs() -> list[tuple[str, str]]:
    return [
        ("c_hcf_lcm", "hard"),
        ("c_trigonometric_ratios", "easy"),
        ("c_classical_probability", "medium"),
    ]
