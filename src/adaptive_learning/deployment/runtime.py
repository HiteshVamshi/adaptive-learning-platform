from __future__ import annotations

from pathlib import Path

from adaptive_learning.deployment.build_pipeline import run_full_build
from adaptive_learning.deployment.settings import DeploymentConfig, DeploymentStatus


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

    run_full_build(paths=paths, config=config)
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
        paths.artifacts_dir / "learn_resources.csv",
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
