"""Single orchestration path for full artifact builds (used by runtime and tests)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from adaptive_learning.content_generation.pipeline import build_content_bundle
from adaptive_learning.data.generator import generate_subject_dataset
from adaptive_learning.data.io import write_dataset
from adaptive_learning.data.sqlite_store import write_dataset_to_sqlite
from adaptive_learning.deployment.settings import DeploymentConfig
from adaptive_learning.fine_tuning.pipeline import run_fine_tuning_pipeline
from adaptive_learning.mastery.pipeline import run_mastery_pipeline
from adaptive_learning.rag.build_index import build_rag_index
from adaptive_learning.recommendation.pipeline import run_recommendation_pipeline
from adaptive_learning.learn.build_index import write_learn_index_for_paths
from adaptive_learning.search.build_index import build_search_index


def run_full_build(*, paths, config: DeploymentConfig) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.search_index_dir.mkdir(parents=True, exist_ok=True)
    paths.rag_index_dir.mkdir(parents=True, exist_ok=True)
    paths.mastery_dir.mkdir(parents=True, exist_ok=True)
    paths.recommendation_dir.mkdir(parents=True, exist_ok=True)
    paths.generated_content_dir.mkdir(parents=True, exist_ok=True)
    paths.fine_tuning_dir.mkdir(parents=True, exist_ok=True)
    paths.agent_trace_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_subject_dataset(paths.subject)
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
        concept_pairs=_default_fine_tuning_pairs(dataset.concepts),
    )
    write_build_manifest(paths=paths, config=config)
    write_learn_index_for_paths(paths)


def write_build_manifest(*, paths, config: DeploymentConfig) -> None:
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "subject": paths.subject,
        "embedding_backend": config.embedding_backend,
        "sqlite_filename": config.sqlite_filename,
        "config": asdict(config),
    }
    out = paths.artifacts_dir / "build_manifest.json"
    with out.open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2)


def _default_generation_concepts(concepts_df) -> list[str]:
    concept_rows = concepts_df[concepts_df["node_type"] == "concept"].copy()
    preferred_ids = [
        "c_hcf_lcm",
        "c_trigonometric_ratios",
        "c_classical_probability",
        "c_acids_bases_ph",
        "c_lenses_refraction",
        "c_ohms_law_resistance",
    ]
    available = concept_rows["concept_id"].astype(str).tolist()
    selected = [concept_id for concept_id in preferred_ids if concept_id in available]
    if len(selected) >= 3:
        return selected[:3]
    fallback = concept_rows.sort_values(by=["chapter_name", "order_index"]).head(3)
    return fallback["concept_id"].astype(str).tolist()


def _default_fine_tuning_pairs(concepts_df) -> list[tuple[str, str]]:
    concept_rows = concepts_df[concepts_df["node_type"] == "concept"].copy()
    if concept_rows.empty:
        return []

    difficulty_map = {"easy": None, "medium": None, "hard": None}
    for row in concept_rows.sort_values(by=["chapter_name", "order_index"]).to_dict(orient="records"):
        band = str(row.get("difficulty_band") or "medium")
        if band == "core":
            band = "medium"
        if band in difficulty_map and difficulty_map[band] is None:
            difficulty_map[band] = str(row["concept_id"])

    ordered_pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for difficulty in ["hard", "easy", "medium"]:
        concept_id = difficulty_map.get(difficulty)
        if concept_id and concept_id not in seen:
            ordered_pairs.append((concept_id, difficulty))
            seen.add(concept_id)

    if ordered_pairs:
        return ordered_pairs

    first_three = concept_rows.head(3)["concept_id"].astype(str).tolist()
    fallback_difficulties = ["easy", "medium", "hard"]
    return list(zip(first_three, fallback_difficulties[: len(first_three)]))
