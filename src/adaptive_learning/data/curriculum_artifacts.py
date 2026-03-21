"""Load curriculum tabular data and concept graph once for search/RAG pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.readwrite import json_graph


def load_concepts_and_graph(data_dir: Path) -> tuple[pd.DataFrame, nx.MultiDiGraph]:
    concepts = pd.read_csv(data_dir / "concepts.csv")
    with (data_dir / "concept_graph.json").open("r", encoding="utf-8") as file_obj:
        concept_graph = json_graph.node_link_graph(json.load(file_obj), multigraph=True)
    return concepts, concept_graph
