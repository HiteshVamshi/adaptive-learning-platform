from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from networkx.readwrite import json_graph

from adaptive_learning.agents.pipeline import AgentSuite, build_agent_suite
from adaptive_learning.config import content_generation_backend, rag_generator_backend
from adaptive_learning.content_generation.generator import build_content_generator
from adaptive_learning.content_generation.pipeline import load_generation_context
from adaptive_learning.fine_tuning.adapter import DifficultyTunedGenerator
from adaptive_learning.mastery.engine import ConceptMasteryEngine
from adaptive_learning.mastery.schemas import AttemptRecord
from adaptive_learning.rag.pipeline import RAGEngine
from adaptive_learning.search.hybrid_search import HybridSearchEngine


@dataclass(frozen=True)
class PlatformPaths:
    subject: str
    root_dir: Path
    artifacts_dir: Path
    data_dir: Path
    search_index_dir: Path
    rag_index_dir: Path
    mastery_dir: Path
    recommendation_dir: Path
    generated_content_dir: Path
    fine_tuning_dir: Path
    agent_trace_dir: Path


@dataclass(frozen=True)
class PlatformArtifacts:
    subject: str
    concepts: pd.DataFrame
    questions: pd.DataFrame
    solutions: pd.DataFrame
    syllabus_topics: pd.DataFrame
    textbook_sections: pd.DataFrame
    theory_content: pd.DataFrame
    concept_graph: object
    attempts: pd.DataFrame
    mastery_snapshot: pd.DataFrame
    mastery_history: pd.DataFrame
    recommendations: pd.DataFrame
    generated_questions: pd.DataFrame
    generated_explanations: pd.DataFrame
    generated_summaries: pd.DataFrame
    adaptation_comparisons: pd.DataFrame
    search_summary: dict
    rag_summary: dict
    mastery_summary: dict
    recommendation_summary: dict
    content_generation_summary: dict
    fine_tuning_summary: dict
    fine_tuning_metrics: dict
    dataset_summary: dict
    official_sources: list[dict[str, str]]
    db_status: dict[str, object]
    agent_traces: dict[str, dict]


@dataclass(frozen=True)
class PlatformServices:
    search_engine: HybridSearchEngine
    rag_engine: RAGEngine
    agent_suite: AgentSuite
    mastery_engine: ConceptMasteryEngine
    content_generator: object
    tuned_generator: DifficultyTunedGenerator | None


def default_paths(root_dir: Path, subject: str = "math") -> PlatformPaths:
    normalized_subject = subject.strip().lower()
    subject_root = root_dir / "artifacts" / "subjects" / normalized_subject
    return PlatformPaths(
        subject=normalized_subject,
        root_dir=root_dir,
        artifacts_dir=subject_root,
        data_dir=subject_root / "bootstrap_data",
        search_index_dir=subject_root / "search_index",
        rag_index_dir=subject_root / "rag_index",
        mastery_dir=subject_root / "mastery",
        recommendation_dir=subject_root / "recommendations",
        generated_content_dir=subject_root / "generated_content",
        fine_tuning_dir=subject_root / "fine_tuning",
        agent_trace_dir=subject_root / "agent_traces",
    )


def load_platform_artifacts(*, paths: PlatformPaths) -> PlatformArtifacts:
    with (paths.data_dir / "concept_graph.json").open("r", encoding="utf-8") as file_obj:
        concept_graph = json_graph.node_link_graph(json.load(file_obj), multigraph=True)

    return PlatformArtifacts(
        subject=paths.subject,
        concepts=pd.read_csv(paths.data_dir / "concepts.csv"),
        questions=pd.read_csv(paths.data_dir / "questions.csv"),
        solutions=pd.read_csv(paths.data_dir / "solutions.csv"),
        syllabus_topics=pd.read_csv(paths.data_dir / "syllabus_topics.csv"),
        textbook_sections=pd.read_csv(paths.data_dir / "textbook_sections.csv"),
        theory_content=pd.read_csv(paths.data_dir / "theory_content.csv"),
        concept_graph=concept_graph,
        attempts=pd.read_csv(paths.mastery_dir / "attempts.csv"),
        mastery_snapshot=pd.read_csv(paths.mastery_dir / "mastery_snapshot.csv"),
        mastery_history=pd.read_csv(paths.mastery_dir / "mastery_history.csv"),
        recommendations=pd.read_csv(paths.recommendation_dir / "recommendations.csv"),
        generated_questions=pd.read_csv(paths.generated_content_dir / "generated_questions.csv"),
        generated_explanations=pd.read_csv(paths.generated_content_dir / "generated_explanations.csv"),
        generated_summaries=pd.read_csv(paths.generated_content_dir / "generated_summaries.csv"),
        adaptation_comparisons=pd.read_csv(paths.fine_tuning_dir / "adaptation_comparisons.csv"),
        search_summary=_read_json(paths.search_index_dir / "index_summary.json"),
        rag_summary=_read_json(paths.rag_index_dir / "rag_summary.json"),
        mastery_summary=_read_json(paths.mastery_dir / "mastery_summary.json"),
        recommendation_summary=_read_json(paths.recommendation_dir / "recommendation_summary.json"),
        content_generation_summary=_read_json(paths.generated_content_dir / "content_generation_summary.json"),
        fine_tuning_summary=_read_json(paths.fine_tuning_dir / "fine_tuning_summary.json"),
        fine_tuning_metrics=_read_json(paths.fine_tuning_dir / "difficulty_metrics.json"),
        dataset_summary=_read_json(paths.data_dir / "dataset_summary.json"),
        official_sources=_official_sources(paths.subject),
        db_status=_load_db_status(paths.data_dir / "adaptive_learning.db"),
        agent_traces=_load_agent_traces(paths.agent_trace_dir),
    )


def build_platform_services(*, paths: PlatformPaths, artifacts: PlatformArtifacts) -> PlatformServices:
    search_engine = HybridSearchEngine.from_artifacts(
        data_dir=paths.data_dir,
        index_dir=paths.search_index_dir,
    )
    rag_engine = RAGEngine.from_artifacts(
        data_dir=paths.data_dir,
        index_dir=paths.rag_index_dir,
        generator_backend=rag_generator_backend(),
    )
    agent_suite = build_agent_suite(
        data_dir=paths.data_dir,
        search_index_dir=paths.search_index_dir,
        rag_index_dir=paths.rag_index_dir,
        mastery_dir=paths.mastery_dir,
        recommendation_dir=paths.recommendation_dir,
    )
    mastery_engine = ConceptMasteryEngine(
        concepts=artifacts.concepts,
        concept_graph=artifacts.concept_graph,
    )
    generation_context = load_generation_context(data_dir=paths.data_dir)
    content_generator = build_content_generator(
        context=generation_context,
        backend=content_generation_backend(),
    )

    tuned_generator = None
    calibrator_path = paths.fine_tuning_dir / "difficulty_calibrator.joblib"
    if calibrator_path.exists():
        tuned_generator = DifficultyTunedGenerator.from_artifacts(
            data_dir=paths.data_dir,
            fine_tuning_dir=paths.fine_tuning_dir,
        )

    return PlatformServices(
        search_engine=search_engine,
        rag_engine=rag_engine,
        agent_suite=agent_suite,
        mastery_engine=mastery_engine,
        content_generator=content_generator,
        tuned_generator=tuned_generator,
    )


def build_practice_attempt(
    *,
    concept_id: str,
    concept_name: str,
    chapter_name: str,
    difficulty: str,
    correct: bool,
    response_time_sec: int,
    user_id: str,
    simulation_step: int,
) -> pd.DataFrame:
    """Single attempt row for a generated practice question (no question_id in bank)."""
    import pandas as pd
    from adaptive_learning.mastery.schemas import AttemptRecord

    ts = pd.Timestamp.utcnow().floor("s").isoformat()
    qid = f"q_gen_{concept_id}_{simulation_step:04d}"
    record = AttemptRecord(
        attempt_id=f"{user_id}_practice_{simulation_step:04d}",
        user_id=user_id,
        simulation_step=simulation_step,
        timestamp=ts,
        question_id=qid,
        concept_id=concept_id,
        concept_name=concept_name,
        chapter_name=chapter_name,
        difficulty=difficulty,
        correct=int(correct),
        response_time_sec=response_time_sec,
        expected_time_sec=90,
        source="streamlit_practice",
    )
    return pd.DataFrame([record.to_dict()])


def build_manual_attempts(
    *,
    questions: pd.DataFrame,
    evaluations: list[dict],
    start_step: int,
    user_id: str,
) -> pd.DataFrame:
    question_lookup = questions.set_index("question_id")
    timestamp = pd.Timestamp.utcnow().floor("s")
    attempts: list[dict] = []

    for offset, evaluation in enumerate(evaluations, start=1):
        question_id = str(evaluation["question_id"])
        question = question_lookup.loc[question_id]
        record = AttemptRecord(
            attempt_id=f"{user_id}_manual_{start_step + offset:03d}",
            user_id=user_id,
            simulation_step=start_step + offset,
            timestamp=(timestamp + pd.Timedelta(minutes=offset)).isoformat(),
            question_id=question_id,
            concept_id=str(question["concept_id"]),
            concept_name=str(question["concept_name"]),
            chapter_name=str(question["chapter_name"]),
            difficulty=str(question["difficulty"]),
            correct=int(bool(evaluation["correct"])),
            response_time_sec=int(evaluation["response_time_sec"]),
            expected_time_sec=int(question["estimated_time_sec"]),
            source="streamlit_test",
        )
        attempts.append(record.to_dict())

    return pd.DataFrame(attempts)


def compute_live_mastery_snapshot(
    *,
    base_attempts: pd.DataFrame,
    manual_attempts: pd.DataFrame,
    mastery_engine: ConceptMasteryEngine,
    fallback_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    if manual_attempts.empty:
        return fallback_snapshot.copy()
    combined_attempts = pd.concat([base_attempts, manual_attempts], ignore_index=True)
    return mastery_engine.compute_snapshot(combined_attempts)


def build_empty_mastery_snapshot(concepts: pd.DataFrame, reference_snapshot: pd.DataFrame) -> pd.DataFrame:
    """Fallback snapshot with zero mastery when user has no attempts."""
    concept_rows = concepts[concepts["node_type"] == "concept"].copy()
    if concept_rows.empty:
        return reference_snapshot.head(0)
    records = []
    for _, row in concept_rows.iterrows():
        records.append({
            "user_id": "",
            "concept_id": str(row["concept_id"]),
            "concept_name": str(row["name"]),
            "chapter_name": str(row["chapter_name"]),
            "attempts_count": 0,
            "correct_count": 0,
            "weighted_accuracy": 0.0,
            "completion": 0.0,
            "speed": 0.0,
            "challenge": 0.0,
            "confidence": 0.0,
            "direct_mastery": 0.0,
            "graph_support": 0.0,
            "graph_adjusted_mastery": 0.0,
            "mastery_band": "needs_support",
            "explanation": "No attempts recorded yet.",
        })
    return pd.DataFrame(records)


def target_difficulty_for_band(mastery_band: str) -> str:
    mapping = {
        "needs_support": "easy",
        "developing": "medium",
        "mastered": "hard",
    }
    return mapping.get(str(mastery_band), "medium")


def build_chapter_coverage_table(*, artifacts: PlatformArtifacts) -> pd.DataFrame:
    syllabus_chapters = (
        artifacts.syllabus_topics[["chapter_id", "chapter_name", "unit_name", "unit_marks", "periods"]]
        .drop_duplicates(subset=["chapter_id"])
        .copy()
    )
    concept_counts = (
        artifacts.concepts[artifacts.concepts["node_type"] == "concept"]
        .groupby("chapter_id")["concept_id"]
        .count()
        .rename("concept_count")
    )
    question_counts = (
        artifacts.questions.groupby("chapter_id")["question_id"].count().rename("question_count")
    )
    theory_counts = (
        artifacts.theory_content.groupby("chapter_id")["content_id"].count().rename("theory_count")
    )
    section_counts = (
        artifacts.textbook_sections.groupby("chapter_id")["section_id"].count().rename("textbook_section_count")
    )
    source_coverage = (
        artifacts.questions.groupby("chapter_id")["source"]
        .apply(lambda values: " | ".join(sorted({str(value) for value in values})))
        .rename("question_sources")
    )

    coverage = syllabus_chapters.merge(
        concept_counts,
        left_on="chapter_id",
        right_index=True,
        how="left",
    )
    coverage = coverage.merge(question_counts, left_on="chapter_id", right_index=True, how="left")
    coverage = coverage.merge(theory_counts, left_on="chapter_id", right_index=True, how="left")
    coverage = coverage.merge(section_counts, left_on="chapter_id", right_index=True, how="left")
    coverage = coverage.merge(source_coverage, left_on="chapter_id", right_index=True, how="left")

    for column in ["concept_count", "question_count", "theory_count", "textbook_section_count"]:
        coverage[column] = coverage[column].fillna(0).astype(int)

    coverage["question_sources"] = coverage["question_sources"].fillna("none")
    coverage["coverage_status"] = coverage.apply(_coverage_status, axis=1)
    return coverage.sort_values(by=["chapter_name"]).reset_index(drop=True)


def _load_agent_traces(trace_dir: Path) -> dict[str, dict]:
    traces = {}
    for path in sorted(trace_dir.glob("*.json")):
        traces[path.stem] = _read_json(path)
    return traces


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _official_sources(subject: str) -> list[dict[str, str]]:
    if subject == "science":
        return [
            {
                "label": "CBSE Class X Science Syllabus 2025-26",
                "kind": "syllabus",
                "document": "CBSE Science Subject Code 086 Classes IX and X (2025-26)",
                "url": "https://cbseacademic.nic.in/web_material/CurriculumMain26/Sec/Science_Sec_2025-26.pdf",
                "usage": "Canonical chapter scope, official topic wording, unit marks, and periods.",
            },
            {
                "label": "NCERT Class X Science Textbook Contents",
                "kind": "textbook_index",
                "document": "NCERT Science Textbook for Class X (Contents)",
                "url": "https://ncert.nic.in/textbook/pdf/jesc1ps.pdf",
                "usage": "Canonical chapter decomposition used to infer official textbook-backed concept coverage.",
            },
            {
                "label": "NCERT Class X Science Textbook Index Page",
                "kind": "textbook_portal",
                "document": "NCERT Textbook Portal",
                "url": "https://ncert.nic.in/textbook.php?jesc1=ps-13",
                "usage": "Human-readable entry point for textbook contents and chapter PDFs.",
            },
        ]
    return [
        {
            "label": "CBSE Class X Mathematics Syllabus 2024-25",
            "kind": "syllabus",
            "document": "CBSE Mathematics (IX-X) Code No. 041 Session 2024-25",
            "url": "https://cbseacademic.nic.in/web_material/CurriculumMain25/Sec/Maths_Sec_2024-25.pdf",
            "usage": "Canonical chapter scope, official topic wording, unit marks, and periods.",
        },
        {
            "label": "NCERT Class X Mathematics Textbook Contents",
            "kind": "textbook_index",
            "document": "NCERT Mathematics Textbook for Class X (Contents)",
            "url": "https://ncert.nic.in/textbook/pdf/jemh1ps.pdf",
            "usage": "Canonical chapter and section decomposition used to infer textbook-backed concept coverage.",
        },
        {
            "label": "NCERT Class X Mathematics Textbook Index Page",
            "kind": "textbook_portal",
            "document": "NCERT Textbook Portal",
            "url": "https://ncert.nic.in/textbook.php?jemh1=ps-14",
            "usage": "Human-readable entry point for textbook contents and chapter PDFs.",
        },
    ]


def _load_db_status(sqlite_path: Path) -> dict[str, object]:
    if not sqlite_path.exists():
        return {
            "available": False,
            "sqlite_path": str(sqlite_path),
            "table_counts": {},
            "metadata": {},
        }

    with sqlite3.connect(sqlite_path) as connection:
        tables = pd.read_sql_query(
            """
            select name
            from sqlite_master
            where type = 'table' and name not like 'sqlite_%'
            order by name
            """,
            connection,
        )["name"].tolist()

        table_counts: dict[str, int] = {}
        for table_name in tables:
            row_count = pd.read_sql_query(
                f"select count(*) as row_count from {table_name}",
                connection,
            ).iloc[0]["row_count"]
            table_counts[str(table_name)] = int(row_count)

        metadata = {}
        if "dataset_metadata" in tables:
            metadata_df = pd.read_sql_query(
                "select key, value from dataset_metadata order by key",
                connection,
            )
            for row in metadata_df.to_dict(orient="records"):
                metadata[str(row["key"])] = json.loads(str(row["value"]))

    return {
        "available": True,
        "sqlite_path": str(sqlite_path),
        "table_counts": table_counts,
        "metadata": metadata,
    }


def _coverage_status(row: pd.Series) -> str:
    if int(row["question_count"]) > 0 and int(row["theory_count"]) > 0:
        return "covered"
    if int(row["question_count"]) > 0 or int(row["theory_count"]) > 0:
        return "partial"
    return "missing"
