from __future__ import annotations

import networkx as nx
import pandas as pd


def build_concept_graph(
    *, concepts: pd.DataFrame, relationships: pd.DataFrame
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(name="cbse_class_10_math_concepts")

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
