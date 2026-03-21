from __future__ import annotations

import networkx as nx
import pandas as pd


def build_concept_graph(
    *,
    concepts: pd.DataFrame,
    relationships: pd.DataFrame,
    subject_key: str = "math",
    class_level: str = "10",
) -> nx.MultiDiGraph:
    safe_subject = str(subject_key).strip().lower().replace(" ", "_") or "unknown"
    graph = nx.MultiDiGraph(name=f"cbse_class_{class_level}_{safe_subject}_concepts")

    for concept in concepts.to_dict(orient="records"):
        concept_id = concept["concept_id"]
        attrs = {key: value for key, value in concept.items() if key != "concept_id"}
        graph.add_node(concept_id, **attrs)

        parent_concept_id = concept.get("parent_concept_id")
        if pd.notna(parent_concept_id) and str(parent_concept_id).strip():
            graph.add_edge(
                parent_concept_id,
                concept_id,
                key=f"contains::{parent_concept_id}::{concept_id}",
                relation_type="contains",
                weight=1.0,
                rationale="Chapter-to-concept hierarchy.",
            )

    for relationship in relationships.to_dict(orient="records"):
        graph.add_edge(
            relationship["source_concept_id"],
            relationship["target_concept_id"],
            key=(
                f"{relationship['relation_type']}::"
                f"{relationship['source_concept_id']}::"
                f"{relationship['target_concept_id']}"
            ),
            relation_type=relationship["relation_type"],
            weight=float(relationship["weight"]),
            rationale=relationship["rationale"],
        )

    return graph


def validate_concept_graph(
    graph: nx.MultiDiGraph,
    *,
    concepts: pd.DataFrame,
    relationships: pd.DataFrame,
) -> list[str]:
    """Return human-readable issues; empty list means no problems detected."""
    issues: list[str] = []
    node_ids = set(concepts["concept_id"].astype(str))
    for row in relationships.to_dict(orient="records"):
        src, tgt = str(row["source_concept_id"]), str(row["target_concept_id"])
        if src not in node_ids:
            issues.append(f"relationship references missing source concept_id={src}")
        if tgt not in node_ids:
            issues.append(f"relationship references missing target concept_id={tgt}")
    for concept in concepts.to_dict(orient="records"):
        cid = str(concept["concept_id"])
        parent = concept.get("parent_concept_id")
        if pd.notna(parent) and str(parent).strip() and str(parent) not in node_ids:
            issues.append(f"concept {cid} parent_concept_id={parent} not in concepts table")
    return issues
