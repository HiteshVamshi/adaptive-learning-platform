from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from adaptive_learning.mastery.schemas import AttemptRecord


DIFFICULTY_MODIFIER = {
    "easy": 0.12,
    "medium": 0.0,
    "hard": -0.16,
}


@dataclass(frozen=True)
class StudentProfile:
    user_id: str
    display_name: str
    concept_skill: dict[str, float]


DEFAULT_STUDENT_PROFILES = {
    "student_cbse_01": StudentProfile(
        user_id="student_cbse_01",
        display_name="Aarav",
        concept_skill={
            "c_euclid_division_lemma": 0.78,
            "c_hcf_lcm": 0.66,
            "c_irrational_numbers": 0.41,
            "c_decimal_expansions": 0.62,
            "c_polynomial_degree_terms": 0.83,
            "c_zeroes_of_polynomial": 0.74,
            "c_zero_coeff_relationship": 0.57,
            "c_factorisation_remainder": 0.48,
            "c_similar_triangles": 0.72,
            "c_basic_proportionality_theorem": 0.55,
            "c_pythagoras_theorem": 0.69,
            "c_similarity_area_ratio": 0.44,
            "c_trigonometric_ratios": 0.64,
            "c_standard_trig_values": 0.82,
            "c_trig_identities": 0.52,
            "c_heights_distances": 0.39,
            "c_distance_formula": 0.77,
            "c_section_formula": 0.58,
            "c_area_of_triangle": 0.49,
        },
    ),
}


class StudentAttemptSimulator:
    def __init__(self, questions: pd.DataFrame, random_seed: int = 7) -> None:
        self.questions = questions.reset_index(drop=True)
        self.rng = np.random.default_rng(random_seed)

    def simulate(
        self,
        *,
        student_profile: StudentProfile,
        num_attempts: int = 60,
        start_timestamp: str = "2026-03-01T09:00:00",
    ) -> pd.DataFrame:
        attempts: list[AttemptRecord] = []
        current_time = pd.Timestamp(start_timestamp)

        seeded_questions = self._seeded_questions_by_concept()
        remaining_attempts = max(0, num_attempts - len(seeded_questions))
        sampled_questions = self._sample_questions(student_profile, remaining_attempts)
        full_sequence = seeded_questions + sampled_questions

        for step, question in enumerate(full_sequence, start=1):
            concept_id = str(question["concept_id"])
            base_skill = student_profile.concept_skill.get(concept_id, 0.5)
            difficulty = str(question["difficulty"])
            success_probability = np.clip(
                base_skill + DIFFICULTY_MODIFIER[difficulty] + self.rng.normal(0.0, 0.08),
                0.05,
                0.98,
            )
            correct = int(self.rng.random() <= success_probability)

            expected_time_sec = int(question["estimated_time_sec"])
            speed_multiplier = np.clip(
                1.35 - base_skill + (0.18 if difficulty == "hard" else 0.0) + self.rng.normal(0.0, 0.12),
                0.55,
                1.95,
            )
            response_time_sec = max(25, int(expected_time_sec * speed_multiplier))

            attempts.append(
                AttemptRecord(
                    attempt_id=f"{student_profile.user_id}_attempt_{step:03d}",
                    user_id=student_profile.user_id,
                    simulation_step=step,
                    timestamp=current_time.isoformat(),
                    question_id=str(question["question_id"]),
                    concept_id=concept_id,
                    concept_name=str(question["concept_name"]),
                    chapter_name=str(question["chapter_name"]),
                    difficulty=difficulty,
                    correct=correct,
                    response_time_sec=response_time_sec,
                    expected_time_sec=expected_time_sec,
                )
            )
            current_time += pd.Timedelta(minutes=int(self.rng.integers(18, 42)))

        return pd.DataFrame([attempt.to_dict() for attempt in attempts])

    def _seeded_questions_by_concept(self) -> list[pd.Series]:
        seeded_questions: list[pd.Series] = []
        for _, group in self.questions.groupby("concept_id", sort=True):
            preferred = group.sort_values(
                by=["difficulty", "estimated_time_sec", "question_id"],
                ascending=[True, True, True],
            ).iloc[0]
            seeded_questions.append(preferred)
        return seeded_questions

    def _sample_questions(self, student_profile: StudentProfile, remaining_attempts: int) -> list[pd.Series]:
        if remaining_attempts <= 0:
            return []
        question_weights = self._question_weights(student_profile)
        question_indices = np.arange(len(self.questions))
        sampled_indices = self.rng.choice(
            question_indices,
            size=remaining_attempts,
            p=question_weights / question_weights.sum(),
        )
        return [self.questions.iloc[int(index)] for index in sampled_indices]

    def _question_weights(self, student_profile: StudentProfile) -> np.ndarray:
        weights = []
        for question in self.questions.to_dict(orient="records"):
            concept_id = str(question["concept_id"])
            difficulty = str(question["difficulty"])
            skill = student_profile.concept_skill.get(concept_id, 0.5)
            weakness_weight = 1.15 - 0.75 * skill
            difficulty_weight = {"easy": 0.9, "medium": 1.0, "hard": 1.05}[difficulty]
            weights.append(max(0.05, weakness_weight * difficulty_weight))
        return np.array(weights, dtype=float)
