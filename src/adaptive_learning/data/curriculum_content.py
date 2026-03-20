from __future__ import annotations

from adaptive_learning.data.cbse_class10_syllabus import (
    CBSE_MATH_SOURCE_DOCUMENT,
    CBSE_MATH_SOURCE_URL,
)
from adaptive_learning.data.schemas import TheoryContentRecord


CONCEPT_TO_SECTION_PATTERNS = {
    "c_fundamental_theorem_arithmetic": ["1.2"],
    "c_euclid_division_lemma": ["1.1"],
    "c_hcf_lcm": ["1.2"],
    "c_irrational_numbers": ["1.3"],
    "c_decimal_expansions": ["1.4"],
    "c_polynomial_degree_terms": ["2.1"],
    "c_zeroes_of_polynomial": ["2.2"],
    "c_zero_coeff_relationship": ["2.3"],
    "c_factorisation_remainder": ["2.4"],
    "c_pair_linear_equations_graphical": ["3.2"],
    "c_pair_linear_equations_consistency": ["3.2", "3.3"],
    "c_pair_linear_equations_algebraic": ["3.3"],
    "c_quadratic_factorization": ["4.2", "4.3"],
    "c_quadratic_formula": ["4.2"],
    "c_discriminant_nature_roots": ["4.4"],
    "c_quadratic_applications": ["4.2", "4.3", "4.4"],
    "c_ap_nth_term": ["5.3"],
    "c_ap_sum": ["5.4"],
    "c_ap_applications": ["5.3", "5.4"],
    "c_distance_formula": ["7.2"],
    "c_section_formula": ["7.3"],
    "c_area_of_triangle": ["7.2", "7.3"],
    "c_similar_triangles": ["6.2", "6.3"],
    "c_basic_proportionality_theorem": ["6.3", "6.4"],
    "c_pythagoras_theorem": ["7.2", "8.2"],
    "c_similarity_area_ratio": ["6.4"],
    "c_tangent_radius_perpendicular": ["10.2"],
    "c_tangents_from_external_point": ["10.3"],
    "c_trigonometric_ratios": ["8.2"],
    "c_standard_trig_values": ["8.3"],
    "c_trig_identities": ["8.4"],
    "c_heights_distances": ["9.1"],
    "c_sectors_segments": ["11.1"],
    "c_circle_perimeter_area_applications": ["11.1"],
    "c_combination_solids_surface_area": ["12.2"],
    "c_combination_solids_volume": ["12.3"],
    "c_grouped_data_mean": ["13.2"],
    "c_grouped_data_median_mode": ["13.3", "13.4"],
    "c_classical_probability": ["14.1"],
}

CHAPTER_FALLBACK_SUMMARY = (
    "This chapter is included from the official CBSE Class X syllabus and aligned with the NCERT "
    "Class X Mathematics textbook contents."
)


def build_theory_content(*, concepts_df, syllabus_topics_df, textbook_sections_df):
    chapter_topic_summary = (
        syllabus_topics_df.groupby(["chapter_id", "chapter_name"])["official_text"]
        .apply(lambda values: " ".join(dict.fromkeys(str(value) for value in values)))
        .to_dict()
    )
    section_lookup = textbook_sections_df.copy()

    theory_rows = []
    concept_rows = concepts_df[concepts_df["node_type"] == "concept"].copy()
    for concept in concept_rows.to_dict(orient="records"):
        chapter_key = (concept["chapter_id"], concept["chapter_name"])
        official_text = chapter_topic_summary.get(chapter_key, "")
        matched_sections = _match_sections(
            concept_id=str(concept["concept_id"]),
            chapter_id=str(concept["chapter_id"]),
            textbook_sections_df=section_lookup,
        )
        section_titles = [str(row["section_title"]) for row in matched_sections]
        section_ids = [str(row["section_id"]) for row in matched_sections]
        section_phrase = ", ".join(section_titles) if section_titles else "the chapter sequence"
        summary_text = (
            f"{concept['name']} belongs to the NCERT Class X chapter {concept['chapter_name']}. "
            f"The official syllabus expects students to {str(concept['learning_objective']).rstrip('.').lower()}. "
            f"In the NCERT textbook, this aligns with {section_phrase}. "
            f"{official_text or CHAPTER_FALLBACK_SUMMARY} "
            f"Core concept description: {concept['description']}"
        )
        theory_rows.append(
            TheoryContentRecord(
                content_id=f"curriculum_note::{concept['concept_id']}",
                class_level=str(concept["class_level"]),
                chapter_id=str(concept["chapter_id"]),
                chapter_name=str(concept["chapter_name"]),
                concept_id=str(concept["concept_id"]),
                concept_name=str(concept["name"]),
                title=f"Theory Note: {concept['name']}",
                content_type="syllabus_textbook_grounded",
                summary_text=summary_text,
                official_syllabus_text=official_text,
                textbook_sections=section_titles,
                textbook_section_ids=section_ids,
                source_document=CBSE_MATH_SOURCE_DOCUMENT,
                source_url=CBSE_MATH_SOURCE_URL,
            ).to_dict()
        )
    return theory_rows


def _match_sections(*, concept_id: str, chapter_id: str, textbook_sections_df):
    candidate_rows = textbook_sections_df[textbook_sections_df["chapter_id"] == chapter_id].copy()
    wanted_prefixes = CONCEPT_TO_SECTION_PATTERNS.get(concept_id, [])
    if wanted_prefixes:
        matched = candidate_rows[
            candidate_rows["section_number"].map(lambda value: any(str(value).startswith(prefix) for prefix in wanted_prefixes))
        ]
        if not matched.empty:
            return matched.to_dict(orient="records")
    return candidate_rows.head(2).to_dict(orient="records")
