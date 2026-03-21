from adaptive_learning.learn.sources.math_programmatic import math_learn_resource_seed
from adaptive_learning.learn.sources.science_programmatic import science_learn_resource_seed

__all__ = ["math_learn_resource_seed", "science_learn_resource_seed"]


def seed_for_subject(subject: str) -> list:
    s = subject.strip().lower()
    if s == "science":
        return science_learn_resource_seed()
    return math_learn_resource_seed()
