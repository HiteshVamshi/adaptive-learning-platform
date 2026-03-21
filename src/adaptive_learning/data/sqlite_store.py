from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from adaptive_learning.data.generator import GeneratedDataset


def write_dataset_to_sqlite(*, dataset: GeneratedDataset, sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        dataset.concepts.to_sql("concepts", connection, if_exists="replace", index=False)
        dataset.questions.to_sql("questions", connection, if_exists="replace", index=False)
        dataset.solutions.to_sql("solutions", connection, if_exists="replace", index=False)
        dataset.relationships.to_sql("concept_relationships", connection, if_exists="replace", index=False)
        dataset.syllabus_topics.to_sql("syllabus_topics", connection, if_exists="replace", index=False)
        dataset.textbook_sections.to_sql("textbook_sections", connection, if_exists="replace", index=False)
        dataset.theory_content.to_sql("theory_content", connection, if_exists="replace", index=False)

        _create_indexes(connection)
        _write_metadata(connection, dataset)


def _create_indexes(connection: sqlite3.Connection) -> None:
    statements = [
        "create index if not exists idx_concepts_concept_id on concepts(concept_id)",
        "create index if not exists idx_questions_question_id on questions(question_id)",
        "create index if not exists idx_questions_concept_id on questions(concept_id)",
        "create index if not exists idx_questions_source on questions(source)",
        "create index if not exists idx_solutions_question_id on solutions(question_id)",
        "create index if not exists idx_theory_content_concept_id on theory_content(concept_id)",
        "create index if not exists idx_textbook_sections_chapter_id on textbook_sections(chapter_id)",
    ]
    cursor = connection.cursor()
    for statement in statements:
        cursor.execute(statement)
    connection.commit()


def _write_metadata(connection: sqlite3.Connection, dataset: GeneratedDataset) -> None:
    cursor = connection.cursor()
    cursor.execute("drop table if exists dataset_metadata")
    cursor.execute("create table dataset_metadata (key text primary key, value text not null)")
    metadata = {
        "subject": dataset.subject,
        "concept_count": len(dataset.concepts),
        "question_count": len(dataset.questions),
        "question_sources": dataset.questions["source"].value_counts().to_dict(),
        "theory_content_count": len(dataset.theory_content),
        "textbook_section_count": len(dataset.textbook_sections),
    }
    for key, value in metadata.items():
        cursor.execute(
            "insert into dataset_metadata(key, value) values (?, ?)",
            (str(key), json.dumps(value)),
        )
    connection.commit()
