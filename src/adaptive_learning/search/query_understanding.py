from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

from adaptive_learning.search.text_utils import normalize_text, significant_tokens


MANUAL_ALIASES = {
    "bpt": "c_basic_proportionality_theorem",
    "basic proportionality theorem": "c_basic_proportionality_theorem",
    "euclid lemma": "c_euclid_division_lemma",
    "euclid division lemma": "c_euclid_division_lemma",
    "hcf": "c_hcf_lcm",
    "lcm": "c_hcf_lcm",
    "pythagoras": "c_pythagoras_theorem",
    "trig": "ch_trigonometry",
    "trigo": "ch_trigonometry",
    "trigonometry": "ch_trigonometry",
    "trig identities": "c_trig_identities",
    "trigonometric identities": "c_trig_identities",
    "trigonometry identities": "c_trig_identities",
    "trig ratios": "c_trigonometric_ratios",
    "trigonometry ratios": "c_trigonometric_ratios",
    "trigonometric ratios": "c_trigonometric_ratios",
    "coordinate geometry": "ch_coordinate_geometry",
    "real numbers": "ch_real_numbers",
    "ph": "c_acids_bases_ph",
    "acid base": "c_acids_bases_ph",
    "acids bases": "c_acids_bases_ph",
    "ohm law": "c_ohms_law_resistance",
    "ohms law": "c_ohms_law_resistance",
    "electric power": "c_electric_power_heating",
    "human eye": "ch_human_eye_colourful_world",
    "light refraction": "ch_light_reflection_refraction",
    "life processes": "ch_life_processes",
    "heredity": "ch_heredity",
    "reproduction": "ch_reproduction",
    "ecosystem": "c_ecosystem_food_chains",
    "food chain": "c_ecosystem_food_chains",
    "biodegradable": "c_biodegradable_ozone",
    "ozone": "c_biodegradable_ozone",
    "domestic circuits": "c_domestic_circuits_ac_dc",
    "magnetic effects": "ch_magnetic_effects_current",
    "chemical reactions": "ch_chemical_reactions",
    "carbon compounds": "ch_carbon_compounds",
}

GENERIC_TAG_ALIASES = {
    "algorithm",
    "angles",
    "application",
    "area",
    "classification",
    "coordinates",
    "distance",
    "evaluation",
    "geometry",
    "proof",
    "ratio",
    "ratios",
    "similarity",
}


@dataclass(frozen=True)
class QueryIntent:
    original_query: str
    normalized_query: str
    requested_difficulty: str | None
    detected_concept_ids: list[str]
    expanded_concept_ids: list[str]
    expanded_terms: list[str]


class QueryUnderstandingEngine:
    def __init__(self, concepts: pd.DataFrame, concept_graph: nx.MultiDiGraph) -> None:
        self.concepts = concepts.copy()
        self.concept_graph = concept_graph
        self.alias_to_concept = self._build_alias_index()

    def understand(self, query: str) -> QueryIntent:
        normalized_query = normalize_text(query)
        requested_difficulty = self.detect_difficulty(normalized_query)
        detected_concept_ids = self.detect_concepts(normalized_query)
        expanded_concept_ids = self.expand_concepts(detected_concept_ids)
        expanded_terms = self.expand_terms(expanded_concept_ids)

        return QueryIntent(
            original_query=query,
            normalized_query=normalized_query,
            requested_difficulty=requested_difficulty,
            detected_concept_ids=detected_concept_ids,
            expanded_concept_ids=expanded_concept_ids,
            expanded_terms=expanded_terms,
        )

    def detect_difficulty(self, query: str) -> str | None:
        difficulty_aliases = {
            "easy": {"easy", "simple", "basic", "beginner"},
            "medium": {"medium", "moderate", "intermediate"},
            "hard": {"hard", "challenging", "advanced", "difficult", "tough"},
        }
        tokens = set(significant_tokens(query))
        for difficulty, aliases in difficulty_aliases.items():
            if tokens.intersection(aliases):
                return difficulty
        return None

    def detect_concepts(self, query: str) -> list[str]:
        matches: list[str] = []
        token_set = set(significant_tokens(query))

        for alias, alias_meta in self.alias_to_concept.items():
            alias_tokens = set(significant_tokens(alias))
            if not alias_tokens:
                continue

            source = alias_meta["source"]
            matched = False
            if source == "manual":
                matched = alias in query or alias_tokens.issubset(token_set)
            else:
                matched = alias in query or len(alias_tokens) >= 2 and alias_tokens.issubset(token_set)

            if matched:
                matches.append(alias_meta["concept_id"])

        return list(dict.fromkeys(matches))

    def expand_concepts(self, concept_ids: list[str]) -> list[str]:
        expanded: list[str] = list(concept_ids)
        for concept_id in concept_ids:
            if concept_id not in self.concept_graph:
                continue

            outgoing = self.concept_graph.out_edges(concept_id, data=True)
            incoming = self.concept_graph.in_edges(concept_id, data=True)

            for _, target_id, edge_data in outgoing:
                if edge_data.get("relation_type") in {"prerequisite", "related", "contains"}:
                    expanded.append(target_id)
            for source_id, _, edge_data in incoming:
                if edge_data.get("relation_type") in {"prerequisite", "related", "contains"}:
                    expanded.append(source_id)

        return list(dict.fromkeys(expanded))

    def expand_terms(self, concept_ids: list[str]) -> list[str]:
        terms: list[str] = []
        for concept_id in concept_ids:
            if concept_id not in self.concept_graph.nodes:
                continue
            node = self.concept_graph.nodes[concept_id]
            name = node.get("name")
            chapter_name = node.get("chapter_name")
            tags = node.get("tags", "")
            if name:
                terms.append(str(name))
            if chapter_name:
                terms.append(str(chapter_name))
            if tags:
                if isinstance(tags, str):
                    terms.extend(tag.strip() for tag in tags.split("|") if tag.strip())
        return list(dict.fromkeys(terms))

    def _build_alias_index(self) -> dict[str, dict[str, str]]:
        alias_index = {
            normalize_text(alias): {"concept_id": concept_id, "source": "manual"}
            for alias, concept_id in MANUAL_ALIASES.items()
        }
        for concept in self.concepts.to_dict(orient="records"):
            concept_id = concept["concept_id"]

            name_alias = normalize_text(str(concept.get("name", "")))
            chapter_alias = normalize_text(str(concept.get("chapter_name", "")))
            for alias in {name_alias, chapter_alias}:
                if alias:
                    alias_index[alias] = {"concept_id": concept_id, "source": "name"}

            tags = concept.get("tags", "")
            if isinstance(tags, str):
                for tag in tags.split("|"):
                    alias = normalize_text(tag)
                    if not alias or alias in GENERIC_TAG_ALIASES:
                        continue
                    if len(significant_tokens(alias)) >= 2:
                        alias_index[alias] = {"concept_id": concept_id, "source": "tag"}
        return alias_index
