from __future__ import annotations

from dataclasses import dataclass

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
