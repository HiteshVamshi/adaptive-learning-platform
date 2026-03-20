from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adaptive_learning.data.schemas import QuestionRecord, SolutionRecord


@dataclass(frozen=True)
class TextbookBootstrapContext:
    concepts: pd.DataFrame
    existing_questions: pd.DataFrame
    theory_content: pd.DataFrame


def generate_textbook_bootstrap_questions(
    *,
    concepts: pd.DataFrame,
    existing_questions: pd.DataFrame,
    theory_content: pd.DataFrame,
) -> tuple[list[QuestionRecord], list[SolutionRecord]]:
    context = TextbookBootstrapContext(
        concepts=concepts[concepts["node_type"] == "concept"].copy(),
        existing_questions=existing_questions.copy(),
        theory_content=theory_content.copy(),
    )
    covered_concepts = set(context.existing_questions["concept_id"].astype(str).tolist())

    question_records: list[QuestionRecord] = []
    solution_records: list[SolutionRecord] = []

    for concept in context.concepts.to_dict(orient="records"):
        concept_id = str(concept["concept_id"])
        if concept_id in covered_concepts:
            continue

        generator = QUESTION_GENERATORS.get(concept_id, _generic_question_spec)
        spec = generator(concept=concept, context=context)
        question_records.append(
            QuestionRecord(
                question_id=f"q_tb_{concept_id.removeprefix('c_')}_001",
                chapter_id=str(concept["chapter_id"]),
                chapter_name=str(concept["chapter_name"]),
                concept_id=concept_id,
                concept_name=str(concept["name"]),
                difficulty=spec["difficulty"],
                question_type=spec["question_type"],
                prompt=spec["prompt"],
                answer=spec["answer"],
                tags=spec["tags"],
                estimated_time_sec=spec["estimated_time_sec"],
                source="textbook_bootstrap",
            )
        )
        solution_records.append(
            SolutionRecord(
                question_id=f"q_tb_{concept_id.removeprefix('c_')}_001",
                final_answer=spec["answer"],
                worked_solution=spec["worked_solution"],
                explanation_style="textbook_bootstrap",
            )
        )

    return question_records, solution_records


def _spec(
    *,
    difficulty: str,
    question_type: str,
    prompt: str,
    answer: str,
    worked_solution: str,
    tags: list[str],
    estimated_time_sec: int,
) -> dict:
    return {
        "difficulty": difficulty,
        "question_type": question_type,
        "prompt": prompt,
        "answer": answer,
        "worked_solution": worked_solution,
        "tags": tags + ["textbook_bootstrap"],
        "estimated_time_sec": estimated_time_sec,
    }


def _fundamental_theorem_arithmetic(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="numerical",
        prompt="Use prime factorisation to find the HCF and LCM of 72 and 120.",
        answer="HCF = 24, LCM = 360",
        worked_solution="Prime factorise: 72 = 2^3 x 3^2 and 120 = 2^3 x 3 x 5. The HCF is the product of the smallest powers of common primes: 2^3 x 3 = 24. The LCM is the product of the greatest powers of all involved primes: 2^3 x 3^2 x 5 = 360.",
        tags=["real_numbers", "prime_factorization", "hcf_lcm"],
        estimated_time_sec=180,
    )


def _pair_linear_graphical(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="easy",
        question_type="graphical",
        prompt="Solve the pair of linear equations x + y = 5 and x - y = 1 graphically. State the point of intersection.",
        answer="(3, 2)",
        worked_solution="The two lines intersect where both equations are satisfied. Adding the equations gives 2x = 6, so x = 3. Substituting into x + y = 5 gives y = 2. Therefore the lines intersect at (3, 2).",
        tags=["linear_equations", "graphical_method"],
        estimated_time_sec=180,
    )


def _pair_linear_consistency(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="reasoning",
        prompt="Determine the number of solutions of the pair 2x + 4y - 6 = 0 and x + 2y - 3 = 0.",
        answer="Infinitely many solutions",
        worked_solution="For a1/a2 = 2/1 = 2, b1/b2 = 4/2 = 2, and c1/c2 = -6/-3 = 2. Since all three ratios are equal, the two equations represent the same line. Hence the pair has infinitely many solutions.",
        tags=["linear_equations", "consistency"],
        estimated_time_sec=150,
    )


def _pair_linear_algebraic(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="algebraic",
        prompt="Solve the pair of equations 2x + y = 11 and x - y = 1 by the elimination method.",
        answer="x = 4, y = 3",
        worked_solution="Add the equations: (2x + y) + (x - y) = 11 + 1, so 3x = 12 and x = 4. Substitute into x - y = 1 to get 4 - y = 1, hence y = 3.",
        tags=["linear_equations", "elimination"],
        estimated_time_sec=150,
    )


def _quadratic_factorization(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="easy",
        question_type="algebraic",
        prompt="Solve x^2 - 9x + 20 = 0 by factorisation.",
        answer="x = 4 or x = 5",
        worked_solution="Factorise x^2 - 9x + 20 as (x - 4)(x - 5) = 0. Therefore x = 4 or x = 5.",
        tags=["quadratic_equations", "factorisation"],
        estimated_time_sec=120,
    )


def _quadratic_formula(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="algebraic",
        prompt="Solve 2x^2 - 7x + 3 = 0 using the quadratic formula.",
        answer="x = 3 or x = 1/2",
        worked_solution="Here a = 2, b = -7, c = 3. Using x = [-b ± sqrt(b^2 - 4ac)]/(2a), x = [7 ± sqrt(49 - 24)]/4 = [7 ± 5]/4. Hence x = 3 or x = 1/2.",
        tags=["quadratic_equations", "quadratic_formula"],
        estimated_time_sec=180,
    )


def _discriminant(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="reasoning",
        prompt="Find the discriminant of 3x^2 - 5x + 2 = 0 and state the nature of its roots.",
        answer="Discriminant = 1; roots are two distinct real roots",
        worked_solution="For a = 3, b = -5, c = 2, the discriminant is D = b^2 - 4ac = 25 - 24 = 1. Since D is positive, the equation has two distinct real roots.",
        tags=["quadratic_equations", "discriminant"],
        estimated_time_sec=120,
    )


def _quadratic_applications(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="hard",
        question_type="application",
        prompt="The product of two consecutive positive integers is 72. Form the quadratic equation and find the integers.",
        answer="8 and 9",
        worked_solution="Let the smaller integer be x. Then x(x + 1) = 72, so x^2 + x - 72 = 0. Factorising gives (x + 9)(x - 8) = 0, so x = 8 or x = -9. Since the integers are positive, they are 8 and 9.",
        tags=["quadratic_equations", "word_problem"],
        estimated_time_sec=210,
    )


def _ap_nth_term(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="easy",
        question_type="numerical",
        prompt="Find the 15th term of the arithmetic progression 7, 11, 15, 19, ...",
        answer="63",
        worked_solution="Here a = 7 and d = 4. The nth term is a_n = a + (n - 1)d. So a_15 = 7 + 14 x 4 = 63.",
        tags=["arithmetic_progression", "nth_term"],
        estimated_time_sec=90,
    )


def _ap_sum(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="numerical",
        prompt="Find the sum of the first 20 terms of the arithmetic progression 3, 7, 11, 15, ...",
        answer="820",
        worked_solution="Here a = 3, d = 4 and n = 20. The 20th term is 3 + 19 x 4 = 79. Then S_20 = n/2 [2a + (n - 1)d] = 20/2 [6 + 76] = 10 x 82 = 820.",
        tags=["arithmetic_progression", "sum"],
        estimated_time_sec=150,
    )


def _ap_applications(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="application",
        prompt="In an auditorium, the first row has 20 seats, the second row has 22 seats, and each next row has 2 more seats than the previous row. How many seats are there in the first 15 rows?",
        answer="510",
        worked_solution="The number of seats forms an AP with a = 20, d = 2, and n = 15. So S_15 = 15/2 [2(20) + 14 x 2] = 15/2 (40 + 28) = 15/2 x 68 = 510.",
        tags=["arithmetic_progression", "application"],
        estimated_time_sec=180,
    )


def _tangent_radius(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="easy",
        question_type="reasoning",
        prompt="A tangent PT touches a circle with centre O at P. What is the measure of angle OPT?",
        answer="90 degrees",
        worked_solution="The radius through the point of contact is perpendicular to the tangent. Therefore angle OPT = 90 degrees.",
        tags=["circles", "tangent", "radius"],
        estimated_time_sec=60,
    )


def _equal_tangents(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="numerical",
        prompt="From an external point P, two tangents PA and PB are drawn to a circle. If PA = 12 cm, find PB.",
        answer="12 cm",
        worked_solution="Tangents drawn from an external point to a circle are equal in length. Hence PB = PA = 12 cm.",
        tags=["circles", "equal_tangents"],
        estimated_time_sec=60,
    )


def _sectors_segments(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="numerical",
        prompt="Find the area of a sector of radius 7 cm and central angle 90 degrees.",
        answer="49pi/4 square cm",
        worked_solution="Area of a sector = (theta/360) x pi r^2. So the area is (90/360) x pi x 7^2 = 49pi/4 square cm.",
        tags=["areas_related_to_circles", "sector_area"],
        estimated_time_sec=120,
    )


def _circle_applications(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="hard",
        question_type="application",
        prompt="Find the perimeter of a semicircular figure of radius 14 cm, including the diameter. Use pi = 22/7.",
        answer="72 cm",
        worked_solution="Perimeter of the semicircular figure = semicircular arc + diameter = pi r + 2r = 22/7 x 14 + 28 = 44 + 28 = 72 cm.",
        tags=["areas_related_to_circles", "perimeter"],
        estimated_time_sec=150,
    )


def _combined_surface_area(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="numerical",
        prompt="A toy is made by attaching a hemisphere of radius 3 cm on top of a cylinder of the same radius and height 6 cm. Find its external surface area, excluding the base touching the table.",
        answer="54pi square cm",
        worked_solution="External surface area = curved surface area of hemisphere + curved surface area of cylinder = 2pi r^2 + 2pi rh = 2pi x 3^2 + 2pi x 3 x 6 = 18pi + 36pi = 54pi square cm.",
        tags=["surface_areas_volumes", "surface_area"],
        estimated_time_sec=180,
    )


def _combined_volume(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="numerical",
        prompt="A solid is formed by a cylinder of radius 3 cm and height 10 cm with a cone of the same radius and height 4 cm on top. Find its volume.",
        answer="102pi cubic cm",
        worked_solution="Volume = volume of cylinder + volume of cone = pi r^2 h + (1/3)pi r^2 h = pi x 3^2 x 10 + (1/3)pi x 3^2 x 4 = 90pi + 12pi = 102pi cubic cm.",
        tags=["surface_areas_volumes", "volume"],
        estimated_time_sec=180,
    )


def _grouped_mean(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="easy",
        question_type="statistics",
        prompt="Find the mean of the grouped data with classes 0-10, 10-20, 20-30 and frequencies 5, 7, 8.",
        answer="16.5",
        worked_solution="Class marks are 5, 15 and 25. Then sum(fx) = 5x5 + 7x15 + 8x25 = 25 + 105 + 200 = 330. Total frequency is 20. Mean = 330/20 = 16.5.",
        tags=["statistics", "grouped_mean"],
        estimated_time_sec=180,
    )


def _grouped_median_mode(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="medium",
        question_type="statistics",
        prompt="Find the mode of the grouped data: class intervals 0-10, 10-20, 20-30, 30-40, 40-50 with frequencies 3, 7, 12, 8, 5.",
        answer="25.56 approximately",
        worked_solution="The modal class is 20-30 because it has the highest frequency 12. Using Mode = l + [(f1 - f0)/(2f1 - f0 - f2)] x h with l = 20, h = 10, f1 = 12, f0 = 7 and f2 = 8, Mode = 20 + (5/9) x 10 = 25.56 approximately.",
        tags=["statistics", "grouped_mode"],
        estimated_time_sec=210,
    )


def _probability(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    return _spec(
        difficulty="easy",
        question_type="probability",
        prompt="A fair die is thrown once. Find the probability of getting a number greater than 4.",
        answer="1/3",
        worked_solution="Possible outcomes are 1, 2, 3, 4, 5, 6. Numbers greater than 4 are 5 and 6, so favorable outcomes = 2. Therefore probability = 2/6 = 1/3.",
        tags=["probability", "classical_definition"],
        estimated_time_sec=90,
    )


def _generic_question_spec(*, concept: dict, context: TextbookBootstrapContext) -> dict:
    theory_row = context.theory_content[context.theory_content["concept_id"] == str(concept["concept_id"])]
    section_hint = ""
    if not theory_row.empty:
        section_hint = str(theory_row.iloc[0]["textbook_sections"]).replace("|", ", ")
    prompt = (
        f"Explain the main idea of {concept['name']} and solve one short textbook-aligned exercise based on {section_hint or concept['chapter_name']}."
    )
    answer = str(concept["learning_objective"])
    worked_solution = (
        f"{concept['name']} in {concept['chapter_name']} focuses on {concept['description']} "
        f"The expected learning outcome is to {str(concept['learning_objective']).rstrip('.').lower()}."
    )
    return _spec(
        difficulty="medium",
        question_type="conceptual",
        prompt=prompt,
        answer=answer,
        worked_solution=worked_solution,
        tags=[str(concept["chapter_name"]).lower().replace(" ", "_"), "generic"],
        estimated_time_sec=150,
    )


QUESTION_GENERATORS = {
    "c_fundamental_theorem_arithmetic": _fundamental_theorem_arithmetic,
    "c_pair_linear_equations_graphical": _pair_linear_graphical,
    "c_pair_linear_equations_consistency": _pair_linear_consistency,
    "c_pair_linear_equations_algebraic": _pair_linear_algebraic,
    "c_quadratic_factorization": _quadratic_factorization,
    "c_quadratic_formula": _quadratic_formula,
    "c_discriminant_nature_roots": _discriminant,
    "c_quadratic_applications": _quadratic_applications,
    "c_ap_nth_term": _ap_nth_term,
    "c_ap_sum": _ap_sum,
    "c_ap_applications": _ap_applications,
    "c_tangent_radius_perpendicular": _tangent_radius,
    "c_tangents_from_external_point": _equal_tangents,
    "c_sectors_segments": _sectors_segments,
    "c_circle_perimeter_area_applications": _circle_applications,
    "c_combination_solids_surface_area": _combined_surface_area,
    "c_combination_solids_volume": _combined_volume,
    "c_grouped_data_mean": _grouped_mean,
    "c_grouped_data_median_mode": _grouped_median_mode,
    "c_classical_probability": _probability,
}
