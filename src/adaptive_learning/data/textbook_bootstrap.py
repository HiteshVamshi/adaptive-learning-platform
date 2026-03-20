from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import pandas as pd

from adaptive_learning.data.schemas import QuestionRecord, SolutionRecord


@dataclass(frozen=True)
class TextbookBootstrapContext:
    concepts: pd.DataFrame
    existing_questions: pd.DataFrame
    theory_content: pd.DataFrame


@dataclass(frozen=True)
class QuestionSpec:
    difficulty: str
    question_type: str
    prompt: str
    answer: str
    worked_solution: str
    tags: list[str]
    estimated_time_sec: int


AUGMENT_EXISTING_CONCEPTS = {
    "c_euclid_division_lemma",
    "c_hcf_lcm",
    "c_irrational_numbers",
    "c_decimal_expansions",
    "c_polynomial_degree_terms",
    "c_zeroes_of_polynomial",
    "c_zero_coeff_relationship",
    "c_factorisation_remainder",
    "c_similar_triangles",
    "c_basic_proportionality_theorem",
    "c_pythagoras_theorem",
    "c_similarity_area_ratio",
    "c_trigonometric_ratios",
    "c_standard_trig_values",
    "c_trig_identities",
    "c_heights_distances",
    "c_distance_formula",
    "c_section_formula",
    "c_area_of_triangle",
}


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
    question_records: list[QuestionRecord] = []
    solution_records: list[SolutionRecord] = []

    for concept in context.concepts.to_dict(orient="records"):
        concept_id = str(concept["concept_id"])
        concept_existing = context.existing_questions[
            context.existing_questions["concept_id"].astype(str) == concept_id
        ]
        has_existing_questions = not concept_existing.empty
        should_augment = concept_id in AUGMENT_EXISTING_CONCEPTS

        if has_existing_questions and not should_augment:
            continue

        generator = QUESTION_GENERATORS.get(concept_id, _generic_question_specs)
        specs = generator(concept=concept, context=context)
        for index, spec in enumerate(specs, start=1):
            question_id = f"q_tb_{concept_id.removeprefix('c_')}_{index:03d}"
            question_records.append(
                QuestionRecord(
                    question_id=question_id,
                    chapter_id=str(concept["chapter_id"]),
                    chapter_name=str(concept["chapter_name"]),
                    concept_id=concept_id,
                    concept_name=str(concept["name"]),
                    difficulty=spec.difficulty,
                    question_type=spec.question_type,
                    prompt=spec.prompt,
                    answer=spec.answer,
                    tags=spec.tags,
                    estimated_time_sec=spec.estimated_time_sec,
                    source="textbook_bootstrap",
                )
            )
            solution_records.append(
                SolutionRecord(
                    question_id=question_id,
                    final_answer=spec.answer,
                    worked_solution=spec.worked_solution,
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
) -> QuestionSpec:
    return QuestionSpec(
        difficulty=difficulty,
        question_type=question_type,
        prompt=prompt,
        answer=answer,
        worked_solution=worked_solution,
        tags=tags + ["textbook_bootstrap"],
        estimated_time_sec=estimated_time_sec,
    )


def _concept_section_text(*, concept_id: str, context: TextbookBootstrapContext) -> str:
    theory_rows = context.theory_content[context.theory_content["concept_id"] == concept_id]
    if theory_rows.empty:
        return ""
    return str(theory_rows.iloc[0]["textbook_sections"]).replace("|", ", ")


def _generic_question_specs(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    section_hint = _concept_section_text(concept_id=str(concept["concept_id"]), context=context)
    chapter_tag = str(concept["chapter_name"]).lower().replace(" ", "_")
    learning_objective = str(concept["learning_objective"]).rstrip(".")
    description = str(concept["description"]).rstrip(".")
    section_phrase = section_hint or str(concept["chapter_name"])

    return [
        _spec(
            difficulty="easy",
            question_type="conceptual",
            prompt=(
                f"State the main idea of {concept['name']} from the NCERT section {section_phrase} and "
                f"name one standard situation where it is applied."
            ),
            answer=learning_objective,
            worked_solution=(
                f"{concept['name']} is used to {learning_objective.lower()}. "
                f"In the textbook context {section_phrase}, the concept focuses on {description.lower()}."
            ),
            tags=[chapter_tag, "generic", "concept_recall"],
            estimated_time_sec=120,
        ),
        _spec(
            difficulty="medium",
            question_type="conceptual",
            prompt=(
                f"Explain why {concept['name']} is important in {concept['chapter_name']} and describe "
                f"the textbook idea highlighted in {section_phrase}."
            ),
            answer=learning_objective,
            worked_solution=(
                f"The concept is important because it helps learners {learning_objective.lower()}. "
                f"The NCERT section {section_phrase} frames it through {description.lower()}."
            ),
            tags=[chapter_tag, "generic", "textbook_grounded"],
            estimated_time_sec=150,
        ),
    ]


def _fundamental_theorem_arithmetic(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="numerical",
            prompt="Use prime factorisation to find the HCF and LCM of 72 and 120.",
            answer="HCF = 24, LCM = 360",
            worked_solution="Prime factorise: 72 = 2^3 x 3^2 and 120 = 2^3 x 3 x 5. The HCF is the product of the smallest powers of common primes: 2^3 x 3 = 24. The LCM is the product of the greatest powers of all involved primes: 2^3 x 3^2 x 5 = 360.",
            tags=["real_numbers", "prime_factorization", "hcf_lcm"],
            estimated_time_sec=180,
        ),
        _spec(
            difficulty="hard",
            question_type="reasoning",
            prompt="Why does the product of two positive integers have a unique prime factorisation apart from the order of the prime factors?",
            answer="Because every composite number can be broken into primes uniquely up to order.",
            worked_solution="The Fundamental Theorem of Arithmetic states that every composite number can be expressed as a product of primes, and this factorisation is unique except for the order of the primes. This is why prime factorisation gives consistent HCF and LCM results.",
            tags=["real_numbers", "fundamental_theorem", "reasoning"],
            estimated_time_sec=210,
        ),
    ]


def _pair_linear_graphical(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="easy",
            question_type="graphical",
            prompt="Solve the pair of linear equations x + y = 5 and x - y = 1 graphically. State the point of intersection.",
            answer="(3, 2)",
            worked_solution="The two lines intersect where both equations are satisfied. Adding the equations gives 2x = 6, so x = 3. Substituting into x + y = 5 gives y = 2. Therefore the lines intersect at (3, 2).",
            tags=["linear_equations", "graphical_method"],
            estimated_time_sec=180,
        ),
        _spec(
            difficulty="medium",
            question_type="graphical",
            prompt="For the equations 2x + y = 7 and x - y = 2, find the solution and explain what the intersection point represents on the graph.",
            answer="(3, 1); it is the common point satisfying both equations.",
            worked_solution="Add the equations to eliminate y: 3x = 9, so x = 3. Then 3 - y = 2 gives y = 1. On the graph, the lines meet at (3, 1), which is the ordered pair satisfying both equations simultaneously.",
            tags=["linear_equations", "graphical_method", "intersection_meaning"],
            estimated_time_sec=180,
        ),
    ]


def _pair_linear_consistency(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="reasoning",
            prompt="Determine the number of solutions of the pair 2x + 4y - 6 = 0 and x + 2y - 3 = 0.",
            answer="Infinitely many solutions",
            worked_solution="For a1/a2 = 2/1 = 2, b1/b2 = 4/2 = 2, and c1/c2 = -6/-3 = 2. Since all three ratios are equal, the two equations represent the same line. Hence the pair has infinitely many solutions.",
            tags=["linear_equations", "consistency"],
            estimated_time_sec=150,
        ),
        _spec(
            difficulty="medium",
            question_type="reasoning",
            prompt="Classify the pair 3x + 2y = 8 and 6x + 4y = 9 as consistent or inconsistent. Also state the number of solutions.",
            answer="Inconsistent; no solution",
            worked_solution="Here a1/a2 = 3/6 = 1/2 and b1/b2 = 2/4 = 1/2, but c1/c2 = 8/9 is not equal to 1/2. So the lines are parallel and distinct. Hence the pair is inconsistent and has no solution.",
            tags=["linear_equations", "consistency", "parallel_lines"],
            estimated_time_sec=150,
        ),
    ]


def _pair_linear_algebraic(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="algebraic",
            prompt="Solve the pair of equations 2x + y = 11 and x - y = 1 by the elimination method.",
            answer="x = 4, y = 3",
            worked_solution="Add the equations: (2x + y) + (x - y) = 11 + 1, so 3x = 12 and x = 4. Substitute into x - y = 1 to get 4 - y = 1, hence y = 3.",
            tags=["linear_equations", "elimination"],
            estimated_time_sec=150,
        ),
        _spec(
            difficulty="hard",
            question_type="application",
            prompt="A test has 20 questions. A correct answer gives 3 marks and a wrong answer deducts 1 mark. If a student attempts all questions and scores 44 marks, how many answers were correct and how many were wrong?",
            answer="16 correct and 4 wrong",
            worked_solution="Let correct answers be x and wrong answers be y. Then x + y = 20 and 3x - y = 44. Adding gives 4x = 64, so x = 16. Then y = 4.",
            tags=["linear_equations", "word_problem", "algebraic_method"],
            estimated_time_sec=210,
        ),
    ]


def _quadratic_factorization(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="easy",
            question_type="algebraic",
            prompt="Solve x^2 - 9x + 20 = 0 by factorisation.",
            answer="x = 4 or x = 5",
            worked_solution="Factorise x^2 - 9x + 20 as (x - 4)(x - 5) = 0. Therefore x = 4 or x = 5.",
            tags=["quadratic_equations", "factorisation"],
            estimated_time_sec=120,
        ),
        _spec(
            difficulty="medium",
            question_type="algebraic",
            prompt="Solve 3x^2 - 13x + 12 = 0 by factorisation.",
            answer="x = 3 or x = 4/3",
            worked_solution="Split the middle term: 3x^2 - 9x - 4x + 12 = 0. Then 3x(x - 3) - 4(x - 3) = 0, so (3x - 4)(x - 3) = 0. Hence x = 4/3 or x = 3.",
            tags=["quadratic_equations", "factorisation", "middle_term_split"],
            estimated_time_sec=180,
        ),
    ]


def _quadratic_formula(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="algebraic",
            prompt="Solve 2x^2 - 7x + 3 = 0 using the quadratic formula.",
            answer="x = 3 or x = 1/2",
            worked_solution="Here a = 2, b = -7, c = 3. Using x = [-b +/- sqrt(b^2 - 4ac)]/(2a), x = [7 +/- sqrt(49 - 24)]/4 = [7 +/- 5]/4. Hence x = 3 or x = 1/2.",
            tags=["quadratic_equations", "quadratic_formula"],
            estimated_time_sec=180,
        ),
        _spec(
            difficulty="hard",
            question_type="algebraic",
            prompt="Solve 6x^2 - x - 2 = 0 using the quadratic formula and simplify the roots.",
            answer="x = 2/3 or x = -1/2",
            worked_solution="For a = 6, b = -1, c = -2, D = 1 + 48 = 49. Therefore x = [1 +/- 7]/12, giving x = 8/12 = 2/3 or x = -6/12 = -1/2.",
            tags=["quadratic_equations", "quadratic_formula", "simplification"],
            estimated_time_sec=210,
        ),
    ]


def _discriminant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="reasoning",
            prompt="Find the discriminant of 3x^2 - 5x + 2 = 0 and state the nature of its roots.",
            answer="Discriminant = 1; roots are two distinct real roots",
            worked_solution="For a = 3, b = -5, c = 2, the discriminant is D = b^2 - 4ac = 25 - 24 = 1. Since D is positive, the equation has two distinct real roots.",
            tags=["quadratic_equations", "discriminant"],
            estimated_time_sec=120,
        ),
        _spec(
            difficulty="medium",
            question_type="reasoning",
            prompt="Without solving the equation, state the nature of roots of 4x^2 + 4x + 1 = 0.",
            answer="Real and equal roots",
            worked_solution="Here D = b^2 - 4ac = 4^2 - 4 x 4 x 1 = 16 - 16 = 0. Therefore the quadratic has real and equal roots.",
            tags=["quadratic_equations", "discriminant", "nature_of_roots"],
            estimated_time_sec=120,
        ),
    ]


def _quadratic_applications(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="hard",
            question_type="application",
            prompt="The product of two consecutive positive integers is 72. Form the quadratic equation and find the integers.",
            answer="8 and 9",
            worked_solution="Let the smaller integer be x. Then x(x + 1) = 72, so x^2 + x - 72 = 0. Factorising gives (x + 9)(x - 8) = 0, so x = 8 or x = -9. Since the integers are positive, they are 8 and 9.",
            tags=["quadratic_equations", "word_problem"],
            estimated_time_sec=210,
        ),
        _spec(
            difficulty="hard",
            question_type="application",
            prompt="A rectangular garden has area 60 m^2 and its length is 7 m more than its breadth. Find its dimensions.",
            answer="Breadth = 5 m, Length = 12 m",
            worked_solution="Let the breadth be x m. Then length = x + 7 and x(x + 7) = 60, so x^2 + 7x - 60 = 0. Factorising gives (x + 12)(x - 5) = 0. So x = 5, since length and breadth are positive. Hence breadth = 5 m and length = 12 m.",
            tags=["quadratic_equations", "word_problem", "geometry_application"],
            estimated_time_sec=240,
        ),
    ]


def _ap_nth_term(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="easy",
            question_type="numerical",
            prompt="Find the 15th term of the arithmetic progression 7, 11, 15, 19, ...",
            answer="63",
            worked_solution="Here a = 7 and d = 4. The nth term is a_n = a + (n - 1)d. So a_15 = 7 + 14 x 4 = 63.",
            tags=["arithmetic_progression", "nth_term"],
            estimated_time_sec=90,
        ),
        _spec(
            difficulty="medium",
            question_type="numerical",
            prompt="Which term of the A.P. 5, 9, 13, 17, ... is 101?",
            answer="25th term",
            worked_solution="Here a = 5 and d = 4. Let the required term be the nth term. Then 101 = 5 + (n - 1)4. So 96 = 4(n - 1), hence n - 1 = 24 and n = 25.",
            tags=["arithmetic_progression", "nth_term", "term_identification"],
            estimated_time_sec=120,
        ),
    ]


def _ap_sum(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="numerical",
            prompt="Find the sum of the first 20 terms of the arithmetic progression 3, 7, 11, 15, ...",
            answer="820",
            worked_solution="Here a = 3, d = 4 and n = 20. The 20th term is 3 + 19 x 4 = 79. Then S_20 = n/2 [2a + (n - 1)d] = 20/2 [6 + 76] = 10 x 82 = 820.",
            tags=["arithmetic_progression", "sum"],
            estimated_time_sec=150,
        ),
        _spec(
            difficulty="medium",
            question_type="numerical",
            prompt="Find the sum of the first 30 multiples of 3.",
            answer="1395",
            worked_solution="The multiples of 3 form an A.P. with a = 3, d = 3, and n = 30. So S_30 = 30/2 [2 x 3 + 29 x 3] = 15(6 + 87) = 15 x 93 = 1395.",
            tags=["arithmetic_progression", "sum", "multiples"],
            estimated_time_sec=150,
        ),
    ]


def _ap_applications(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="application",
            prompt="In an auditorium, the first row has 20 seats, the second row has 22 seats, and each next row has 2 more seats than the previous row. How many seats are there in the first 15 rows?",
            answer="510",
            worked_solution="The number of seats forms an AP with a = 20, d = 2, and n = 15. So S_15 = 15/2 [2(20) + 14 x 2] = 15/2 (40 + 28) = 15/2 x 68 = 510.",
            tags=["arithmetic_progression", "application"],
            estimated_time_sec=180,
        ),
        _spec(
            difficulty="hard",
            question_type="application",
            prompt="A child saves Rs 5 on the first day, Rs 8 on the second day, Rs 11 on the third day, and continues in the same pattern. How much will the child save in 20 days?",
            answer="670 rupees",
            worked_solution="Savings form an A.P. with a = 5, d = 3, and n = 20. Then S_20 = 20/2 [2 x 5 + 19 x 3] = 10(10 + 57) = 670. So the child saves Rs 670 in 20 days.",
            tags=["arithmetic_progression", "application", "daily_life"],
            estimated_time_sec=180,
        ),
    ]


def _tangent_radius(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="easy",
            question_type="reasoning",
            prompt="A tangent PT touches a circle with centre O at P. What is the measure of angle OPT?",
            answer="90 degrees",
            worked_solution="The radius through the point of contact is perpendicular to the tangent. Therefore angle OPT = 90 degrees.",
            tags=["circles", "tangent", "radius"],
            estimated_time_sec=60,
        ),
        _spec(
            difficulty="medium",
            question_type="proof",
            prompt="Explain why the radius drawn to the point of contact of a tangent is perpendicular to the tangent.",
            answer="The radius to the point of contact is perpendicular to the tangent.",
            worked_solution="A tangent touches the circle at exactly one point. The shortest distance from the centre to the tangent is along the perpendicular. Since the radius to the point of contact gives this shortest distance, it is perpendicular to the tangent.",
            tags=["circles", "tangent", "proof"],
            estimated_time_sec=150,
        ),
    ]


def _equal_tangents(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="numerical",
            prompt="From an external point P, two tangents PA and PB are drawn to a circle. If PA = 12 cm, find PB.",
            answer="12 cm",
            worked_solution="Tangents drawn from an external point to a circle are equal in length. Hence PB = PA = 12 cm.",
            tags=["circles", "equal_tangents"],
            estimated_time_sec=60,
        ),
        _spec(
            difficulty="medium",
            question_type="geometry",
            prompt="From a point P outside a circle, tangents PA and PB are drawn. If APB = 70 degrees, find angle AOB, where O is the centre.",
            answer="110 degrees",
            worked_solution="OA is perpendicular to PA and OB is perpendicular to PB. In quadrilateral OAPB, angles at A and B are 90 degrees each. Therefore angle AOB + angle APB + 90 + 90 = 360. So angle AOB = 360 - 250 = 110 degrees.",
            tags=["circles", "equal_tangents", "angles"],
            estimated_time_sec=180,
        ),
    ]


def _sectors_segments(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="numerical",
            prompt="Find the area of a sector of radius 7 cm and central angle 90 degrees.",
            answer="49pi/4 square cm",
            worked_solution="Area of a sector = (theta/360) x pi r^2. So the area is (90/360) x pi x 7^2 = 49pi/4 square cm.",
            tags=["areas_related_to_circles", "sector_area"],
            estimated_time_sec=120,
        ),
        _spec(
            difficulty="hard",
            question_type="numerical",
            prompt="A sector of a circle has radius 14 cm and angle 60 degrees. Find its arc length using pi = 22/7.",
            answer="44/3 cm",
            worked_solution="Arc length = (theta/360) x 2pi r = (60/360) x 2 x 22/7 x 14 = 44/3 cm.",
            tags=["areas_related_to_circles", "arc_length"],
            estimated_time_sec=150,
        ),
    ]


def _circle_applications(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="hard",
            question_type="application",
            prompt="Find the perimeter of a semicircular figure of radius 14 cm, including the diameter. Use pi = 22/7.",
            answer="72 cm",
            worked_solution="Perimeter of the semicircular figure = semicircular arc + diameter = pi r + 2r = 22/7 x 14 + 28 = 44 + 28 = 72 cm.",
            tags=["areas_related_to_circles", "perimeter"],
            estimated_time_sec=150,
        ),
        _spec(
            difficulty="hard",
            question_type="application",
            prompt="A quadrant has radius 14 cm. Find its perimeter using pi = 22/7.",
            answer="50 cm",
            worked_solution="Perimeter of a quadrant = arc length of one-fourth of a circle + two radii = (1/4)(2pi r) + 2r = (1/2)pi r + 2r = 22 + 28 = 50 cm.",
            tags=["areas_related_to_circles", "perimeter", "quadrant"],
            estimated_time_sec=150,
        ),
    ]


def _combined_surface_area(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="numerical",
            prompt="A toy is made by attaching a hemisphere of radius 3 cm on top of a cylinder of the same radius and height 6 cm. Find its external surface area, excluding the base touching the table.",
            answer="54pi square cm",
            worked_solution="External surface area = curved surface area of hemisphere + curved surface area of cylinder = 2pi r^2 + 2pi rh = 2pi x 3^2 + 2pi x 3 x 6 = 18pi + 36pi = 54pi square cm.",
            tags=["surface_areas_volumes", "surface_area"],
            estimated_time_sec=180,
        ),
        _spec(
            difficulty="hard",
            question_type="numerical",
            prompt="A solid metallic object is formed by a cone of radius 3 cm and height 4 cm mounted on a hemisphere of radius 3 cm. Find its total external surface area in terms of pi.",
            answer="33pi square cm",
            worked_solution="Slant height of cone = sqrt(3^2 + 4^2) = 5 cm. External surface area = curved surface area of cone + curved surface area of hemisphere = pi r l + 2pi r^2 = pi x 3 x 5 + 2pi x 3^2 = 15pi + 18pi = 33pi square cm.",
            tags=["surface_areas_volumes", "surface_area", "combined_solids"],
            estimated_time_sec=210,
        ),
    ]


def _combined_volume(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="numerical",
            prompt="A solid is formed by a cylinder of radius 3 cm and height 10 cm with a cone of the same radius and height 4 cm on top. Find its volume.",
            answer="102pi cubic cm",
            worked_solution="Volume = volume of cylinder + volume of cone = pi r^2 h + (1/3)pi r^2 h = pi x 3^2 x 10 + (1/3)pi x 3^2 x 4 = 90pi + 12pi = 102pi cubic cm.",
            tags=["surface_areas_volumes", "volume"],
            estimated_time_sec=180,
        ),
        _spec(
            difficulty="hard",
            question_type="application",
            prompt="A medicine capsule consists of a cylinder of height 14 mm with two hemispherical ends of radius 3 mm. Find its volume in cubic millimetres in terms of pi.",
            answer="162pi cubic mm",
            worked_solution="The two hemispheres together make one sphere of radius 3 mm. Volume = volume of cylinder + volume of sphere = pi x 3^2 x 14 + (4/3)pi x 3^3 = 126pi + 36pi = 162pi cubic mm.",
            tags=["surface_areas_volumes", "volume", "capsule"],
            estimated_time_sec=210,
        ),
    ]


def _grouped_mean(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="easy",
            question_type="statistics",
            prompt="Find the mean of the grouped data with classes 0-10, 10-20, 20-30 and frequencies 5, 7, 8.",
            answer="16.5",
            worked_solution="Class marks are 5, 15 and 25. Then sum(fx) = 5x5 + 7x15 + 8x25 = 25 + 105 + 200 = 330. Total frequency is 20. Mean = 330/20 = 16.5.",
            tags=["statistics", "grouped_mean"],
            estimated_time_sec=180,
        ),
        _spec(
            difficulty="medium",
            question_type="statistics",
            prompt="Find the mean of the grouped data: class intervals 10-20, 20-30, 30-40, 40-50 with frequencies 4, 6, 10, 5.",
            answer="31.4",
            worked_solution="Class marks are 15, 25, 35 and 45. Sum(fx) = 4x15 + 6x25 + 10x35 + 5x45 = 60 + 150 + 350 + 225 = 785. Total frequency = 25. Mean = 785/25 = 31.4.",
            tags=["statistics", "grouped_mean", "class_mark"],
            estimated_time_sec=180,
        ),
    ]


def _grouped_median_mode(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="medium",
            question_type="statistics",
            prompt="Find the mode of the grouped data: class intervals 0-10, 10-20, 20-30, 30-40, 40-50 with frequencies 3, 7, 12, 8, 5.",
            answer="25.56 approximately",
            worked_solution="The modal class is 20-30 because it has the highest frequency 12. Using Mode = l + [(f1 - f0)/(2f1 - f0 - f2)] x h with l = 20, h = 10, f1 = 12, f0 = 7 and f2 = 8, Mode = 20 + (5/9) x 10 = 25.56 approximately.",
            tags=["statistics", "grouped_mode"],
            estimated_time_sec=210,
        ),
        _spec(
            difficulty="medium",
            question_type="statistics",
            prompt="Find the median of the grouped data: class intervals 0-10, 10-20, 20-30, 30-40 with frequencies 5, 9, 7, 4.",
            answer="18.33 approximately",
            worked_solution="Total frequency N = 25, so N/2 = 12.5. Cumulative frequencies are 5, 14, 21, 25, so the median class is 10-20. Using Median = l + [(N/2 - cf)/f] x h with l = 10, cf = 5, f = 9, h = 10, Median = 10 + (7.5/9) x 10 = 18.33 approximately.",
            tags=["statistics", "grouped_median"],
            estimated_time_sec=210,
        ),
    ]


def _probability(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    return [
        _spec(
            difficulty="easy",
            question_type="probability",
            prompt="A fair die is thrown once. Find the probability of getting a number greater than 4.",
            answer="1/3",
            worked_solution="Possible outcomes are 1, 2, 3, 4, 5, 6. Numbers greater than 4 are 5 and 6, so favorable outcomes = 2. Therefore probability = 2/6 = 1/3.",
            tags=["probability", "classical_definition"],
            estimated_time_sec=90,
        ),
        _spec(
            difficulty="easy",
            question_type="probability",
            prompt="A card is drawn from a well-shuffled deck of 52 playing cards. Find the probability that the card drawn is a king.",
            answer="1/13",
            worked_solution="There are 52 equally likely outcomes and 4 favorable outcomes because there are 4 kings in a deck. So the probability is 4/52 = 1/13.",
            tags=["probability", "classical_definition", "cards"],
            estimated_time_sec=120,
        ),
        _spec(
            difficulty="medium",
            question_type="probability",
            prompt="One card is drawn from cards numbered 1 to 20. Find the probability that the number is a multiple of 3 or 5.",
            answer="9/20",
            worked_solution="Multiples of 3 are 3, 6, 9, 12, 15, 18 and multiples of 5 are 5, 10, 15, 20. Counting without repetition gives favorable outcomes 3, 5, 6, 9, 10, 12, 15, 18, 20, that is 9 outcomes. So probability = 9/20.",
            tags=["probability", "classical_definition", "counting"],
            estimated_time_sec=150,
        ),
    ]



def _stable_index(*parts: str, modulo: int) -> int:
    joined = '::'.join(parts)
    digest = hashlib.md5(joined.encode('utf-8')).hexdigest()
    return int(digest[:8], 16) % modulo


def _stable_choice(options: list, *parts: str):
    return options[_stable_index(*parts, modulo=len(options))]


def _stable_int(low: int, high: int, *parts: str) -> int:
    return low + _stable_index(*parts, modulo=(high - low + 1))


def _section_prefix(*, concept: dict, context: TextbookBootstrapContext) -> str:
    section_text = _concept_section_text(concept_id=str(concept["concept_id"]), context=context)
    if not section_text:
        return ''
    return f"Based on the NCERT section {section_text}, "


def _real_numbers_euclid_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    divisor = _stable_choice([18, 21, 24, 27, 31, 35], str(concept['concept_id']), 'divisor')
    quotient = _stable_int(7, 14, str(concept['concept_id']), 'quotient')
    remainder = _stable_int(1, divisor - 1, str(concept['concept_id']), 'remainder')
    dividend = divisor * quotient + remainder
    prefix = _section_prefix(concept=concept, context=context)
    return [
        _spec(
            difficulty='easy',
            question_type='short_answer',
            prompt=(
                f"{prefix}use Euclid's division lemma to express {dividend} in the form {divisor}q + r, "
                f"where 0 <= r < {divisor}. Find q and r."
            ),
            answer=f"q = {quotient}, r = {remainder}",
            worked_solution=(
                f"Divide {dividend} by {divisor}. Since {divisor} x {quotient} = {divisor * quotient} and "
                f"{dividend} - {divisor * quotient} = {remainder}, we get {dividend} = {divisor} x {quotient} + {remainder}."
            ),
            tags=['real_numbers', 'division_lemma', 'textbook_variant'],
            estimated_time_sec=120,
        )
    ]


def _real_numbers_hcf_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    gcd = _stable_choice([6, 8, 9, 12], str(concept['concept_id']), 'gcd')
    pair = _stable_choice([(5, 7), (7, 11), (8, 15), (9, 14)], str(concept['concept_id']), 'pair')
    a, b = gcd * pair[0], gcd * pair[1]
    lcm = (a * b) // gcd
    prefix = _section_prefix(concept=concept, context=context)
    return [
        _spec(
            difficulty='medium',
            question_type='numerical',
            prompt=(
                f"{prefix}the HCF of {a} and {b} is {gcd}. Use HCF x LCM = product of the numbers to find the LCM."
            ),
            answer=str(lcm),
            worked_solution=f"LCM = ({a} x {b}) / {gcd} = {lcm}.",
            tags=['real_numbers', 'hcf_lcm', 'product_relationship'],
            estimated_time_sec=120,
        )
    ]


def _real_numbers_irrational_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    a = _stable_choice([2, 3, 4, 6], str(concept['concept_id']), 'constant')
    n = _stable_choice([2, 5, 7, 11, 13], str(concept['concept_id']), 'surd')
    prefix = _section_prefix(concept=concept, context=context)
    return [
        _spec(
            difficulty='hard',
            question_type='proof',
            prompt=f"{prefix}show that {a} + sqrt({n}) is irrational.",
            answer=f"{a} + sqrt({n}) is irrational.",
            worked_solution=(
                f"Assume {a} + sqrt({n}) is rational. Then subtracting {a} would make sqrt({n}) rational, "
                f"which is impossible because {n} is not a perfect square. Hence {a} + sqrt({n}) is irrational."
            ),
            tags=['real_numbers', 'irrational', 'proof'],
            estimated_time_sec=180,
        )
    ]


def _real_numbers_decimal_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    fraction = _stable_choice([(17, 160, 'terminating'), (11, 75, 'terminating'), (13, 96, 'non-terminating recurring')], str(concept['concept_id']), 'fraction')
    num, den, label = fraction
    reason = 'only 2s and 5s as prime factors' if label == 'terminating' else 'a prime factor other than 2 or 5 in the denominator'
    prefix = _section_prefix(concept=concept, context=context)
    return [
        _spec(
            difficulty='medium',
            question_type='classification',
            prompt=(
                f"{prefix}without long division, determine whether the decimal expansion of {num}/{den} is "
                f"terminating or non-terminating recurring."
            ),
            answer=label,
            worked_solution=(
                f"Write the denominator in lowest form. Since {den} has {reason}, the decimal expansion is {label}."
            ),
            tags=['real_numbers', 'decimal_expansion', 'classification'],
            estimated_time_sec=120,
        )
    ]


def _polynomial_degree_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    lead = _stable_choice([3, 5, 7, 9], str(concept['concept_id']), 'lead')
    power = _stable_choice([4, 5, 6], str(concept['concept_id']), 'power')
    mid_coeff = _stable_choice([2, 4, 6], str(concept['concept_id']), 'mid')
    prompt = f"Identify the degree and leading coefficient of p(x) = {lead}x^{power} - {mid_coeff}x^2 + 11."
    return [
        _spec(
            difficulty='easy',
            question_type='short_answer',
            prompt=prompt,
            answer=f"Degree = {power}, leading coefficient = {lead}",
            worked_solution=(
                f"The highest power of x in the polynomial is {power}, so the degree is {power}. "
                f"The coefficient of x^{power} is {lead}, so that is the leading coefficient."
            ),
            tags=['polynomials', 'degree', 'leading_coefficient'],
            estimated_time_sec=90,
        )
    ]


def _zeroes_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    roots = _stable_choice([(2, -5), (3, 4), (-2, 5), (1, -6)], str(concept['concept_id']), 'roots')
    r1, r2 = roots
    sum_roots = r1 + r2
    product = r1 * r2
    sign = '-' if sum_roots >= 0 else '+'
    middle = abs(sum_roots)
    if product >= 0:
        tail = f'+ {product}'
    else:
        tail = f'- {abs(product)}'
    polynomial = f"x^2 {sign} {middle}x {tail}"
    return [
        _spec(
            difficulty='medium',
            question_type='algebraic',
            prompt=f"Find the zeroes of {polynomial}.",
            answer=f"{r1} and {r2}",
            worked_solution=(
                f"The quadratic factors as (x - ({r1}))(x - ({r2})) = 0. Therefore the zeroes are {r1} and {r2}."
            ),
            tags=['polynomials', 'zeroes', 'factorisation'],
            estimated_time_sec=150,
        )
    ]


def _zero_coeff_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    sum_roots, product = _stable_choice([(6, 8), (5, 6), (7, 10), (9, 14)], str(concept['concept_id']), 'sum_product')
    return [
        _spec(
            difficulty='medium',
            question_type='construction',
            prompt=(
                f"Construct a monic quadratic polynomial whose sum of zeroes is {sum_roots} and product of zeroes is {product}."
            ),
            answer=f"x^2 - {sum_roots}x + {product}",
            worked_solution=(
                f"For a monic quadratic, p(x) = x^2 - (sum of zeroes)x + product of zeroes. "
                f"Hence p(x) = x^2 - {sum_roots}x + {product}."
            ),
            tags=['polynomials', 'coefficients', 'construction'],
            estimated_time_sec=120,
        )
    ]


def _remainder_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    root = _stable_choice([2, 3, -1, -2], str(concept['concept_id']), 'root')
    other1, other2 = _stable_choice([(1, -3), (4, -1), (2, -5), (5, -2)], str(concept['concept_id']), 'others')
    value = root
    remainder = (value - root) * (value - other1) * (value - other2)
    polynomial = f"(x - {root})(x - {other1})(x - {other2})"
    divisor_text = f"x - {value}" if value >= 0 else f"x + {abs(value)}"
    return [
        _spec(
            difficulty='hard',
            question_type='verification',
            prompt=(
                f"If p(x) = {polynomial}, find the remainder when p(x) is divided by {divisor_text}."
            ),
            answer=str(remainder),
            worked_solution=(
                f"By the remainder theorem, the remainder on division by {divisor_text} is p({value}). "
                f"Since p({value}) = ({value} - {root})({value} - {other1})({value} - {other2}) = {remainder}, the remainder is {remainder}."
            ),
            tags=['polynomials', 'remainder_theorem', 'textbook_variant'],
            estimated_time_sec=180,
        )
    ]


def _triangles_similarity_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    base = _stable_choice([(4, 6, 8, 2), (5, 12, 13, 3), (6, 8, 10, 4)], str(concept['concept_id']), 'similarity')
    a, b, c, scale = base
    return [
        _spec(
            difficulty='easy',
            question_type='reasoning',
            prompt=(
                f"The sides of one triangle are {a} cm, {b} cm, {c} cm. Another triangle has sides {a * scale} cm, {b * scale} cm, {c * scale} cm. Are the triangles similar?"
            ),
            answer='Yes, the triangles are similar.',
            worked_solution=(
                f"The corresponding side ratios are all {scale}:1. Since the three pairs of corresponding sides are proportional, the triangles are similar by SSS similarity."
            ),
            tags=['triangles', 'similarity', 'sss'],
            estimated_time_sec=120,
        )
    ]


def _triangles_bpt_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    ad = _stable_choice([3, 4, 5], str(concept['concept_id']), 'ad')
    db = _stable_choice([2, 3, 5], str(concept['concept_id']), 'db')
    ae = _stable_choice([4, 6, 7.5], str(concept['concept_id']), 'ae')
    ac = ae * (ad + db) / ad
    return [
        _spec(
            difficulty='medium',
            question_type='numerical',
            prompt=(
                f"In triangle ABC, D lies on AB and E lies on AC. If DE is parallel to BC, AD = {ad} cm, DB = {db} cm, and AE = {ae} cm, find AC."
            ),
            answer=f"{ac:g} cm",
            worked_solution=(
                f"Since DE is parallel to BC, triangles ADE and ABC are similar. Therefore AD/AB = AE/AC. Here AB = {ad + db}. So {ad}/{ad + db} = {ae}/AC, giving AC = {ac:g} cm."
            ),
            tags=['triangles', 'bpt', 'parallel_lines'],
            estimated_time_sec=180,
        )
    ]


def _triangles_pythagoras_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    a, b, c = _stable_choice([(8, 15, 17), (5, 12, 13), (7, 24, 25)], str(concept['concept_id']), 'pythagoras')
    return [
        _spec(
            difficulty='medium',
            question_type='numerical',
            prompt=f"A right triangle has legs {a} cm and {b} cm. Find the hypotenuse.",
            answer=f"{c} cm",
            worked_solution=(
                f"By Pythagoras theorem, hypotenuse = sqrt({a}^2 + {b}^2) = sqrt({a*a} + {b*b}) = sqrt({c*c}) = {c} cm."
            ),
            tags=['triangles', 'pythagoras', 'right_triangle'],
            estimated_time_sec=120,
        )
    ]


def _triangles_area_ratio_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    small, large, area_small = _stable_choice([(2, 5, 18), (3, 4, 27), (4, 7, 32)], str(concept['concept_id']), 'area_ratio')
    area_large = area_small * (large * large) / (small * small)
    return [
        _spec(
            difficulty='hard',
            question_type='application',
            prompt=(
                f"Two similar triangles have corresponding sides in the ratio {small}:{large}. If the area of the smaller triangle is {area_small} cm^2, find the area of the larger triangle."
            ),
            answer=f"{area_large:g} cm^2",
            worked_solution=(
                f"For similar triangles, the ratio of areas equals the square of the ratio of corresponding sides. So smaller:larger area = {small*small}:{large*large}. Therefore the larger area is {area_large:g} cm^2."
            ),
            tags=['triangles', 'similarity', 'area_ratio'],
            estimated_time_sec=180,
        )
    ]


def _trig_ratios_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    opp, adj, hyp = _stable_choice([(5, 12, 13), (7, 24, 25), (8, 15, 17)], str(concept['concept_id']), 'ratios')
    return [
        _spec(
            difficulty='easy',
            question_type='numerical',
            prompt=(
                f"In a right triangle, the side opposite theta is {opp} cm and the adjacent side is {adj} cm. Find sin(theta), cos(theta), and tan(theta)."
            ),
            answer=f"sin(theta) = {opp}/{hyp}, cos(theta) = {adj}/{hyp}, tan(theta) = {opp}/{adj}",
            worked_solution=(
                f"The hypotenuse is {hyp}. Therefore sin(theta) = opposite/hypotenuse = {opp}/{hyp}, cos(theta) = {adj}/{hyp}, and tan(theta) = {opp}/{adj}."
            ),
            tags=['trigonometry', 'ratios', 'right_triangle'],
            estimated_time_sec=150,
        )
    ]


def _trig_standard_values_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    expr, answer, explanation = _stable_choice([
        ('2 sin 30 degrees + cos 60 degrees', '3/2', '2 x 1/2 + 1/2 = 3/2.'),
        ('sin 45 degrees + cos 45 degrees', 'sqrt(2)', 'Each value is 1/sqrt(2), so the sum is 2/sqrt(2) = sqrt(2).'),
        ('tan 45 degrees + sin 30 degrees', '3/2', 'tan 45 degrees = 1 and sin 30 degrees = 1/2, so the sum is 3/2.'),
    ], str(concept['concept_id']), 'std_values')
    return [
        _spec(
            difficulty='easy',
            question_type='short_answer',
            prompt=f"Evaluate {expr}.",
            answer=answer,
            worked_solution=explanation,
            tags=['trigonometry', 'standard_values', 'evaluation'],
            estimated_time_sec=90,
        )
    ]


def _trig_identity_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    num, den = _stable_choice([(3, 4), (5, 12), (8, 15)], str(concept['concept_id']), 'identity')
    sec_sq = 1 + (num * num) / (den * den)
    return [
        _spec(
            difficulty='medium',
            question_type='verification',
            prompt=f"If tan(theta) = {num}/{den}, verify that 1 + tan^2(theta) = sec^2(theta).",
            answer=f"Both sides equal {(den*den + num*num)}/{den*den}",
            worked_solution=(
                f"tan^2(theta) = {num*num}/{den*den}. Hence 1 + tan^2(theta) = ({den*den} + {num*num})/{den*den} = {(den*den + num*num)}/{den*den}. Therefore sec^2(theta) has the same value."
            ),
            tags=['trigonometry', 'identity', 'verification'],
            estimated_time_sec=150,
        )
    ]


def _trig_heights_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    distance, angle, height = _stable_choice([(20, 45, '20 m'), (15, 45, '15 m'), (10, 60, '10*sqrt(3) m')], str(concept['concept_id']), 'height')
    return [
        _spec(
            difficulty='hard',
            question_type='application',
            prompt=(
                f"From a point on the ground {distance} m away from the base of a tower, the angle of elevation of the top is {angle} degrees. Find the height of the tower."
            ),
            answer=height,
            worked_solution=(
                f"Using tan {angle} degrees = height/{distance}, we solve for the height. This gives {height}."
            ),
            tags=['trigonometry', 'heights_distances', 'application'],
            estimated_time_sec=180,
        )
    ]


def _coord_distance_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    x1, y1, dx, dy = _stable_choice([(1, 2, 6, 8), (-2, 1, 5, 12), (3, -1, 8, 15)], str(concept['concept_id']), 'distance')
    x2, y2 = x1 + dx, y1 + dy
    dist = int(math.sqrt(dx * dx + dy * dy))
    return [
        _spec(
            difficulty='easy',
            question_type='numerical',
            prompt=f"Find the distance between ({x1}, {y1}) and ({x2}, {y2}).",
            answer=str(dist),
            worked_solution=(
                f"Distance = sqrt(({x2} - {x1})^2 + ({y2} - {y1})^2) = sqrt({dx}^2 + {dy}^2) = {dist}."
            ),
            tags=['coordinate_geometry', 'distance_formula'],
            estimated_time_sec=90,
        )
    ]


def _coord_section_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    x1, y1, x2, y2, m, n = _stable_choice([(1, 2, 10, 8, 1, 2), (2, 3, 8, 9, 2, 1), (-1, 4, 5, 10, 1, 1)], str(concept['concept_id']), 'section')
    px = (m * x2 + n * x1) / (m + n)
    py = (m * y2 + n * y1) / (m + n)
    return [
        _spec(
            difficulty='medium',
            question_type='numerical',
            prompt=(
                f"Find the point dividing the line segment joining ({x1}, {y1}) and ({x2}, {y2}) internally in the ratio {m}:{n}."
            ),
            answer=f"({px:g}, {py:g})",
            worked_solution=(
                f"Using the section formula, P = (({m} x {x2} + {n} x {x1})/{m+n}, ({m} x {y2} + {n} x {y1})/{m+n}) = ({px:g}, {py:g})."
            ),
            tags=['coordinate_geometry', 'section_formula', 'ratio'],
            estimated_time_sec=150,
        )
    ]


def _coord_area_variant(*, concept: dict, context: TextbookBootstrapContext) -> list[QuestionSpec]:
    base_start, base_end, apex_x, apex_y = _stable_choice([(0, 6, 2, 4), (1, 7, 3, 5), (-2, 4, 1, 3)], str(concept['concept_id']), 'area')
    area = abs(base_end - base_start) * abs(apex_y) / 2
    return [
        _spec(
            difficulty='medium',
            question_type='numerical',
            prompt=(
                f"Find the area of the triangle with vertices ({base_start}, 0), ({base_end}, 0), and ({apex_x}, {apex_y})."
            ),
            answer=f"{area:g} square units",
            worked_solution=(
                f"Take the base on the x-axis. Base length = {abs(base_end - base_start)} and height = {abs(apex_y)}. So area = 1/2 x {abs(base_end - base_start)} x {abs(apex_y)} = {area:g} square units."
            ),
            tags=['coordinate_geometry', 'triangle_area'],
            estimated_time_sec=150,
        )
    ]

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


QUESTION_GENERATORS.update(
    {
        "c_euclid_division_lemma": _real_numbers_euclid_variant,
        "c_hcf_lcm": _real_numbers_hcf_variant,
        "c_irrational_numbers": _real_numbers_irrational_variant,
        "c_decimal_expansions": _real_numbers_decimal_variant,
        "c_polynomial_degree_terms": _polynomial_degree_variant,
        "c_zeroes_of_polynomial": _zeroes_variant,
        "c_zero_coeff_relationship": _zero_coeff_variant,
        "c_factorisation_remainder": _remainder_variant,
        "c_similar_triangles": _triangles_similarity_variant,
        "c_basic_proportionality_theorem": _triangles_bpt_variant,
        "c_pythagoras_theorem": _triangles_pythagoras_variant,
        "c_similarity_area_ratio": _triangles_area_ratio_variant,
        "c_trigonometric_ratios": _trig_ratios_variant,
        "c_standard_trig_values": _trig_standard_values_variant,
        "c_trig_identities": _trig_identity_variant,
        "c_heights_distances": _trig_heights_variant,
        "c_distance_formula": _coord_distance_variant,
        "c_section_formula": _coord_section_variant,
        "c_area_of_triangle": _coord_area_variant,
    }
)
