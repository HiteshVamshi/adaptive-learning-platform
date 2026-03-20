from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from adaptive_learning.content_generation.prompt_templates import (
    explanation_generation_prompt,
    question_generation_prompt,
    summary_generation_prompt,
)
from adaptive_learning.content_generation.schemas import (
    GeneratedExplanationRecord,
    GeneratedQuestionRecord,
    GeneratedSummaryRecord,
)


@dataclass(frozen=True)
class GenerationContext:
    concepts: pd.DataFrame
    questions: pd.DataFrame
    solutions: pd.DataFrame


GERUND_MAP = {
    "apply": "applying",
    "classify": "classifying",
    "compute": "computing",
    "connect": "connecting",
    "define": "defining",
    "express": "expressing",
    "find": "finding",
    "identify": "identifying",
    "model": "modelling",
    "recall": "recalling",
    "recognize": "recognizing",
    "relate": "relating",
    "solve": "solving",
    "use": "using",
}


def _clean_fragment(text: str) -> str:
    return str(text).strip().rstrip(".")


def _metadata_clause(text: str) -> str:
    cleaned = _clean_fragment(text)
    if not cleaned:
        return ""
    parts = cleaned.split(" ", 1)
    first_word = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    if first_word in GERUND_MAP and rest:
        return f"{GERUND_MAP[first_word]} {rest}".lower()
    return cleaned.lower()


class GroundedContentGenerator:
    backend_name = "grounded_templates"

    def __init__(self, context: GenerationContext, random_seed: int = 11) -> None:
        self.context = context
        self.concepts = context.concepts[context.concepts["node_type"] == "concept"].copy()
        self.questions = context.questions.copy()
        self.solutions = context.solutions.copy()
        self.concepts_by_id = self.concepts.set_index("concept_id")
        self.rng = np.random.default_rng(random_seed)

    def generate_question(self, *, concept_id: str, difficulty: str) -> tuple[GeneratedQuestionRecord, str]:
        concept = self.concepts_by_id.loc[concept_id]
        prompt = question_generation_prompt(
            concept_name=str(concept["name"]),
            chapter_name=str(concept["chapter_name"]),
            difficulty=difficulty,
        )
        generated = self._question_for_concept(concept_id=concept_id, difficulty=difficulty, concept=concept)
        return generated, prompt

    def generate_explanation(
        self, *, concept_id: str, audience_level: str = "class_10_student"
    ) -> tuple[GeneratedExplanationRecord, str]:
        concept = self.concepts_by_id.loc[concept_id]
        prompt = explanation_generation_prompt(
            concept_name=str(concept["name"]),
            audience_level=audience_level,
        )
        description = _metadata_clause(concept["description"])
        learning_objective = _clean_fragment(concept["learning_objective"])
        explanation = (
            f"{concept['name']} in {concept['chapter_name']} focuses on {description}. "
            f"A Class 10 student should use this concept to {learning_objective.lower()}."
        )
        key_points = [
            f"Core idea: {_clean_fragment(concept['name'])}.",
            f"Use-case: {learning_objective}.",
            f"Chapter context: {_clean_fragment(concept['chapter_name'])}.",
        ]
        return (
            GeneratedExplanationRecord(
                concept_id=str(concept_id),
                concept_name=str(concept["name"]),
                chapter_name=str(concept["chapter_name"]),
                audience_level=audience_level,
                explanation=explanation,
                key_points=key_points,
                backend=self.backend_name,
            ),
            prompt,
        )

    def generate_summary(
        self, *, scope_type: str, scope_id: str
    ) -> tuple[GeneratedSummaryRecord, str]:
        if scope_type == "chapter":
            chapter_concepts = self.concepts[self.concepts["chapter_id"] == scope_id].copy()
            chapter_name = str(chapter_concepts["chapter_name"].iloc[0])
            prompt = summary_generation_prompt(title=chapter_name)
            bullet_points = [
                f"{row['name']}: {_clean_fragment(row['learning_objective'])}"
                for row in chapter_concepts.sort_values(by="order_index").head(5).to_dict(orient="records")
            ]
            summary = (
                f"{chapter_name} covers {len(chapter_concepts)} connected ideas that move from definition to "
                f"problem solving. Students should be able to identify each concept, choose the right method, "
                f"and explain why the method works."
            )
            return (
                GeneratedSummaryRecord(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    title=chapter_name,
                    summary=summary,
                    bullet_points=bullet_points,
                    backend=self.backend_name,
                ),
                prompt,
            )

        concept = self.concepts_by_id.loc[scope_id]
        prompt = summary_generation_prompt(title=str(concept["name"]))
        summary = (
            f"{concept['name']} is a focused topic inside {concept['chapter_name']}. "
            f"It is mainly about {_metadata_clause(concept['description'])}."
        )
        bullet_points = [
            f"Goal: {_clean_fragment(concept['learning_objective'])}",
            f"Difficulty band: {_clean_fragment(concept['difficulty_band'])}",
            f"Chapter: {_clean_fragment(concept['chapter_name'])}",
        ]
        return (
            GeneratedSummaryRecord(
                scope_type=scope_type,
                scope_id=scope_id,
                title=str(concept["name"]),
                summary=summary,
                bullet_points=bullet_points,
                backend=self.backend_name,
            ),
            prompt,
        )

    def _question_for_concept(
        self, *, concept_id: str, difficulty: str, concept: pd.Series
    ) -> GeneratedQuestionRecord:
        if concept_id in CONCEPT_GENERATORS:
            generator = CONCEPT_GENERATORS[concept_id]
            prompt, final_answer, explanation, tags = generator(self, concept, difficulty)
        else:
            prompt, final_answer, explanation, tags = self._generic_conceptual_question(concept, difficulty)
        return GeneratedQuestionRecord(
            concept_id=str(concept_id),
            concept_name=str(concept["name"]),
            chapter_name=str(concept["chapter_name"]),
            difficulty=difficulty,
            prompt=prompt,
            final_answer=final_answer,
            explanation=explanation,
            tags=tags,
            backend=self.backend_name,
        )

    def _generic_conceptual_question(self, concept: pd.Series, difficulty: str):
        learning_objective = _clean_fragment(concept["learning_objective"])
        description = _metadata_clause(concept["description"])
        prompt = (
            f"Explain the main idea of {concept['name']} and describe one situation where a Class 10 student "
            f"should apply it."
        )
        final_answer = learning_objective
        explanation = (
            f"{concept['name']} is used when we need to {learning_objective.lower()}. "
            f"It belongs to {concept['chapter_name']} and focuses on {description}."
        )
        tags = [str(concept["chapter_name"]).lower().replace(" ", "_"), "generated", difficulty]
        return prompt, final_answer, explanation, tags


def _euclid_division_lemma(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    b = int(generator.rng.integers(17, 43))
    q = int(generator.rng.integers(6, 15))
    r = int(generator.rng.integers(1, b - 1))
    a = b * q + r
    prompt = f"Use Euclid's division lemma to express {a} in the form {b}q + r, where 0 <= r < {b}."
    final_answer = f"q = {q}, r = {r}"
    explanation = f"Divide {a} by {b}. Since {b} x {q} = {b*q} and {a} - {b*q} = {r}, we get {a} = {b} x {q} + {r}."
    return prompt, final_answer, explanation, ["real_numbers", "division_lemma", difficulty]


def _hcf_lcm(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    gcd = int(generator.rng.choice([4, 6, 9, 12]))
    x, y = 7, 11
    a, b = gcd * x, gcd * y
    if difficulty == "hard":
        lcm = (a * b) // gcd
        prompt = f"The HCF of {a} and {b} is {gcd}. Use HCF x LCM = product of the numbers to find the LCM."
        final_answer = str(lcm)
        explanation = f"LCM = ({a} x {b}) / {gcd} = {lcm}."
    else:
        prompt = f"Use Euclid's algorithm to find the HCF of {a} and {b}."
        final_answer = str(gcd)
        explanation = f"The common factor is {gcd}, and Euclid's repeated division reaches {gcd} as the last non-zero remainder."
    return prompt, final_answer, explanation, ["real_numbers", "hcf_lcm", difficulty]


def _decimal_expansion(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    options = [(19, 250, "terminating"), (7, 48, "non-terminating recurring")]
    numerator, denominator, label = options[0] if difficulty != "hard" else options[1]
    prompt = f"Without long division, decide whether {numerator}/{denominator} has a terminating or recurring decimal expansion."
    final_answer = label
    explanation = (
        f"Write the denominator in lowest form. {denominator} has prime factors "
        + ("only 2s and 5s, so the decimal terminates." if label == "terminating" else "including a prime other than 2 or 5, so the decimal recurs.")
    )
    return prompt, final_answer, explanation, ["real_numbers", "decimal_expansion", difficulty]


def _irrational_numbers(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Show that 4 + sqrt(7) is irrational."
    final_answer = "4 + sqrt(7) is irrational."
    explanation = "Assume 4 + sqrt(7) is rational. Subtracting 4 would make sqrt(7) rational, which is impossible. Hence 4 + sqrt(7) is irrational."
    return prompt, final_answer, explanation, ["real_numbers", "irrational", difficulty]


def _polynomial_degree(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Find the degree and leading coefficient of 7x^5 - 2x^2 + 9."
    final_answer = "Degree = 5, leading coefficient = 7"
    explanation = "The highest power of x is 5, so the degree is 5. The coefficient of x^5 is 7."
    return prompt, final_answer, explanation, ["polynomials", "degree", difficulty]


def _zeroes_of_polynomial(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Find the zeroes of x^2 - 3x - 4."
    final_answer = "4 and -1"
    explanation = "Factor the quadratic: x^2 - 3x - 4 = (x - 4)(x + 1). So the zeroes are 4 and -1."
    return prompt, final_answer, explanation, ["polynomials", "zeroes", difficulty]


def _zero_coeff_relationship(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Construct a monic quadratic polynomial whose sum of zeroes is 5 and product of zeroes is 6."
    final_answer = "x^2 - 5x + 6"
    explanation = "For a monic quadratic, p(x) = x^2 - (sum of zeroes)x + product of zeroes. So p(x) = x^2 - 5x + 6."
    return prompt, final_answer, explanation, ["polynomials", "coefficients", difficulty]


def _factorisation_remainder(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Find the remainder when p(x) = x^3 + 2x^2 - x + 5 is divided by x - 2."
    final_answer = "19"
    explanation = "By the remainder theorem, the remainder is p(2) = 8 + 8 - 2 + 5 = 19."
    return prompt, final_answer, explanation, ["polynomials", "remainder_theorem", difficulty]


def _similar_triangles(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "The sides of two triangles are 5 cm, 12 cm, 13 cm and 10 cm, 24 cm, 26 cm. Are the triangles similar?"
    final_answer = "Yes, they are similar."
    explanation = "The corresponding side ratios are 10/5 = 24/12 = 26/13 = 2, so the triangles are similar by SSS similarity."
    return prompt, final_answer, explanation, ["triangles", "similarity", difficulty]


def _bpt(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "In triangle ABC, DE is parallel to BC. If AD = 4 cm, DB = 6 cm, and AE = 5 cm, find AC."
    final_answer = "12.5 cm"
    explanation = "Since DE is parallel to BC, triangles ADE and ABC are similar. So AD/AB = AE/AC. Here AB = 10, so 4/10 = 5/AC and AC = 12.5 cm."
    return prompt, final_answer, explanation, ["triangles", "bpt", difficulty]


def _pythagoras(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "A right triangle has legs 8 cm and 15 cm. Find its hypotenuse."
    final_answer = "17 cm"
    explanation = "By Pythagoras theorem, hypotenuse = sqrt(8^2 + 15^2) = sqrt(64 + 225) = sqrt(289) = 17."
    return prompt, final_answer, explanation, ["triangles", "pythagoras", difficulty]


def _similarity_area_ratio(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Two similar triangles have corresponding sides in the ratio 2:5. If the area of the smaller triangle is 18 cm^2, find the area of the larger triangle."
    final_answer = "112.5 cm^2"
    explanation = "Area ratio equals the square of the side ratio, so it is 4:25. If 4 parts make 18, then 25 parts make 112.5."
    return prompt, final_answer, explanation, ["triangles", "area_ratio", difficulty]


def _trig_ratios(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "In a right triangle, the side opposite theta is 5 cm and the adjacent side is 12 cm. Find sin(theta), cos(theta), and tan(theta)."
    final_answer = "sin(theta) = 5/13, cos(theta) = 12/13, tan(theta) = 5/12"
    explanation = "The hypotenuse is 13 by Pythagoras. Then sin = 5/13, cos = 12/13, and tan = 5/12."
    return prompt, final_answer, explanation, ["trigonometry", "ratios", difficulty]


def _standard_values(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Write the values of sin 60 degrees, cos 30 degrees, and tan 45 degrees."
    final_answer = "sin 60 = sqrt(3)/2, cos 30 = sqrt(3)/2, tan 45 = 1"
    explanation = "These are standard trigonometric values obtained from the 30-60-90 and 45-45-90 triangles."
    return prompt, final_answer, explanation, ["trigonometry", "standard_values", difficulty]


def _trig_identities(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "If sin(theta) = 8/17, find cos(theta) for an acute angle and verify sin^2(theta) + cos^2(theta) = 1."
    final_answer = "cos(theta) = 15/17; identity verified"
    explanation = "For an acute angle, cos(theta) = sqrt(1 - (8/17)^2) = 15/17. Then 64/289 + 225/289 = 1."
    return prompt, final_answer, explanation, ["trigonometry", "identities", difficulty]


def _heights_distances(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "From a point 15 m from the base of a tower, the angle of elevation is 45 degrees. Find the height of the tower."
    final_answer = "15 m"
    explanation = "tan 45 degrees = height / 15 = 1, so the height is 15 m."
    return prompt, final_answer, explanation, ["trigonometry", "heights_distances", difficulty]


def _distance_formula(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Find the distance between (-1, 2) and (5, 10)."
    final_answer = "10"
    explanation = "Distance = sqrt((5 + 1)^2 + (10 - 2)^2) = sqrt(6^2 + 8^2) = sqrt(100) = 10."
    return prompt, final_answer, explanation, ["coordinate_geometry", "distance", difficulty]


def _section_formula(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Find the point dividing the segment joining (2, 4) and (8, 10) internally in the ratio 1:2."
    final_answer = "(4, 6)"
    explanation = "Using the section formula, P = ((1 x 8 + 2 x 2)/3, (1 x 10 + 2 x 4)/3) = (4, 6)."
    return prompt, final_answer, explanation, ["coordinate_geometry", "section_formula", difficulty]


def _area_triangle(generator: GroundedContentGenerator, concept: pd.Series, difficulty: str):
    prompt = "Find the area of the triangle with vertices (1, 1), (5, 1), and (3, 6)."
    final_answer = "10 square units"
    explanation = "Using the coordinate area formula, the area is 1/2 |1(1-6) + 5(6-1) + 3(1-1)| = 10."
    return prompt, final_answer, explanation, ["coordinate_geometry", "triangle_area", difficulty]


CONCEPT_GENERATORS = {
    "c_euclid_division_lemma": _euclid_division_lemma,
    "c_hcf_lcm": _hcf_lcm,
    "c_decimal_expansions": _decimal_expansion,
    "c_irrational_numbers": _irrational_numbers,
    "c_polynomial_degree_terms": _polynomial_degree,
    "c_zeroes_of_polynomial": _zeroes_of_polynomial,
    "c_zero_coeff_relationship": _zero_coeff_relationship,
    "c_factorisation_remainder": _factorisation_remainder,
    "c_similar_triangles": _similar_triangles,
    "c_basic_proportionality_theorem": _bpt,
    "c_pythagoras_theorem": _pythagoras,
    "c_similarity_area_ratio": _similarity_area_ratio,
    "c_trigonometric_ratios": _trig_ratios,
    "c_standard_trig_values": _standard_values,
    "c_trig_identities": _trig_identities,
    "c_heights_distances": _heights_distances,
    "c_distance_formula": _distance_formula,
    "c_section_formula": _section_formula,
    "c_area_of_triangle": _area_triangle,
}


class TransformersContentGenerator:
    backend_name = "transformers"

    def __init__(self, context: GenerationContext, model_name: str = "google/flan-t5-base") -> None:
        from transformers import pipeline

        self.context = context
        self.pipeline = pipeline(
            task="text2text-generation",
            model=model_name,
            tokenizer=model_name,
        )
        self.model_name = model_name
        self.fallback = GroundedContentGenerator(context)

    def generate_question(self, *, concept_id: str, difficulty: str):
        grounded, prompt = self.fallback.generate_question(concept_id=concept_id, difficulty=difficulty)
        output = self.pipeline(prompt, max_new_tokens=160, truncation=True)[0]["generated_text"]
        return grounded.__class__(**{**grounded.__dict__, "explanation": output.strip(), "backend": self.backend_name}), prompt

    def generate_explanation(self, *, concept_id: str, audience_level: str = "class_10_student"):
        grounded, prompt = self.fallback.generate_explanation(concept_id=concept_id, audience_level=audience_level)
        output = self.pipeline(prompt, max_new_tokens=160, truncation=True)[0]["generated_text"]
        return grounded.__class__(**{**grounded.__dict__, "explanation": output.strip(), "backend": self.backend_name}), prompt

    def generate_summary(self, *, scope_type: str, scope_id: str):
        grounded, prompt = self.fallback.generate_summary(scope_type=scope_type, scope_id=scope_id)
        output = self.pipeline(prompt, max_new_tokens=180, truncation=True)[0]["generated_text"]
        return grounded.__class__(**{**grounded.__dict__, "summary": output.strip(), "backend": self.backend_name}), prompt


def build_content_generator(
    *, context: GenerationContext, backend: str = "grounded", model_name: str = "google/flan-t5-base"
):
    if backend == "grounded":
        return GroundedContentGenerator(context)
    if backend == "transformers":
        return TransformersContentGenerator(context, model_name=model_name)
    try:
        return TransformersContentGenerator(context, model_name=model_name)
    except Exception:
        return GroundedContentGenerator(context)
