from __future__ import annotations


def parse_pipe_list(value: str) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def graph_to_dot(
    concept_graph,
    *,
    focus_concept_id: str | None = None,
    chapter_id: str | None = None,
    max_nodes: int = 14,
) -> str:
    included_nodes = set()

    if focus_concept_id and focus_concept_id in concept_graph:
        included_nodes.add(focus_concept_id)
        for source_id, _, edge_data in concept_graph.in_edges(focus_concept_id, data=True):
            if edge_data.get("relation_type") in {"prerequisite", "related"}:
                included_nodes.add(source_id)
        for _, target_id, edge_data in concept_graph.out_edges(focus_concept_id, data=True):
            if edge_data.get("relation_type") in {"prerequisite", "related"}:
                included_nodes.add(target_id)
    elif chapter_id:
        for node_id, node_data in concept_graph.nodes(data=True):
            if node_data.get("chapter_id") == chapter_id or node_id == chapter_id:
                included_nodes.add(node_id)
    else:
        ordered_nodes = sorted(
            concept_graph.nodes(data=True),
            key=lambda item: (
                item[1].get("node_type") != "chapter",
                item[1].get("chapter_name", ""),
                item[1].get("order_index", 0),
                item[1].get("name", ""),
            ),
        )
        for node_id, _ in ordered_nodes[:max_nodes]:
            included_nodes.add(node_id)

    if len(included_nodes) > max_nodes and focus_concept_id:
        concept_nodes = [
            node_id
            for node_id in included_nodes
            if concept_graph.nodes[node_id].get("node_type") == "concept"
        ]
        concept_nodes = sorted(
            concept_nodes,
            key=lambda node_id: concept_graph.nodes[node_id].get("name", ""),
        )
        included_nodes = {focus_concept_id, *concept_nodes[: max_nodes - 1]}

    dot_lines = [
        "digraph ConceptGraph {",
        'rankdir="LR";',
        'graph [bgcolor="white"];',
        'node [shape="box", style="rounded,filled", fontname="Helvetica"];',
        'edge [fontname="Helvetica"];',
    ]

    for node_id in included_nodes:
        node_data = concept_graph.nodes[node_id]
        label = _escape_label(node_data.get("name", node_id))
        if node_data.get("node_type") == "chapter":
            fillcolor = "#d8f3dc"
        elif node_id == focus_concept_id:
            fillcolor = "#ffe8a3"
        else:
            fillcolor = "#e8f1ff"
        dot_lines.append(f'"{node_id}" [label="{label}", fillcolor="{fillcolor}"];')

    for source_id, target_id, edge_data in concept_graph.edges(data=True):
        if source_id not in included_nodes or target_id not in included_nodes:
            continue
        relation_type = edge_data.get("relation_type", "")
        color = "#1d3557" if relation_type == "prerequisite" else "#457b9d"
        label = _escape_label(relation_type)
        dot_lines.append(
            f'"{source_id}" -> "{target_id}" [label="{label}", color="{color}"];'
        )

    dot_lines.append("}")
    return "\n".join(dot_lines)


def mastery_color_map() -> dict[str, str]:
    return {
        "mastered": "#2a9d8f",
        "developing": "#e9c46a",
        "needs_support": "#e76f51",
    }


def _escape_label(value: str) -> str:
    return str(value).replace('"', "'")
