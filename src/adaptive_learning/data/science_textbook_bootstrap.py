from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adaptive_learning.data.schemas import QuestionRecord, SolutionRecord


@dataclass(frozen=True)
class ScienceBootstrapContext:
    concepts: pd.DataFrame
    theory_content: pd.DataFrame


@dataclass(frozen=True)
class ScienceQuestionSpec:
    difficulty: str
    question_type: str
    prompt: str
    answer: str
    worked_solution: str
    tags: list[str]
    estimated_time_sec: int


def generate_science_bootstrap_questions(
    *,
    concepts: pd.DataFrame,
    theory_content: pd.DataFrame,
) -> tuple[list[QuestionRecord], list[SolutionRecord]]:
    context = ScienceBootstrapContext(
        concepts=concepts[concepts["node_type"] == "concept"].copy(),
        theory_content=theory_content.copy(),
    )

    question_records: list[QuestionRecord] = []
    solution_records: list[SolutionRecord] = []

    for concept in context.concepts.to_dict(orient="records"):
        concept_id = str(concept["concept_id"])
        spec = QUESTION_SPECS.get(concept_id, _generic_spec(concept=concept, context=context))
        question_id = f"q_sc_{concept_id.removeprefix('c_')}_001"
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
                tags=spec.tags + ["science_bootstrap"],
                estimated_time_sec=spec.estimated_time_sec,
                source="textbook_bootstrap",
            )
        )
        solution_records.append(
            SolutionRecord(
                question_id=question_id,
                final_answer=spec.answer,
                worked_solution=spec.worked_solution,
                explanation_style="science_bootstrap",
            )
        )

    return question_records, solution_records


def _spec(*, difficulty: str, question_type: str, prompt: str, answer: str, worked_solution: str, tags: list[str], estimated_time_sec: int) -> ScienceQuestionSpec:
    return ScienceQuestionSpec(
        difficulty=difficulty,
        question_type=question_type,
        prompt=prompt,
        answer=answer,
        worked_solution=worked_solution,
        tags=tags,
        estimated_time_sec=estimated_time_sec,
    )


def _generic_spec(*, concept: dict, context: ScienceBootstrapContext) -> ScienceQuestionSpec:
    return _spec(
        difficulty="medium",
        question_type="conceptual",
        prompt=f"Explain the main idea of {concept['name']} and give one NCERT-aligned Class X example.",
        answer=str(concept["learning_objective"]),
        worked_solution=(
            f"{concept['name']} in {concept['chapter_name']} is mainly about {concept['description']} "
            f"The expected learning outcome is to {str(concept['learning_objective']).rstrip('.').lower()}."
        ),
        tags=[str(concept["chapter_name"]).lower().replace(" ", "_"), "conceptual"],
        estimated_time_sec=150,
    )


QUESTION_SPECS = {
    "c_balancing_chemical_equations": _spec(
        difficulty="easy",
        question_type="chemistry",
        prompt="Balance the equation: Fe + H2O -> Fe3O4 + H2.",
        answer="3Fe + 4H2O -> Fe3O4 + 4H2",
        worked_solution="Count atoms on both sides and adjust coefficients until Fe, H and O atoms are equal. The balanced equation is 3Fe + 4H2O -> Fe3O4 + 4H2.",
        tags=["science", "chemistry", "balancing"],
        estimated_time_sec=150,
    ),
    "c_types_chemical_reactions": _spec(
        difficulty="medium",
        question_type="chemistry",
        prompt="Classify the reaction Zn + CuSO4 -> ZnSO4 + Cu and explain your choice.",
        answer="Displacement reaction",
        worked_solution="Zinc displaces copper from copper sulphate solution because zinc is more reactive than copper. So this is a displacement reaction.",
        tags=["science", "chemistry", "reaction_types"],
        estimated_time_sec=120,
    ),
    "c_acids_bases_ph": _spec(
        difficulty="easy",
        question_type="chemistry",
        prompt="A solution has pH 2. Is it acidic or basic? Would blue litmus remain blue or turn red?",
        answer="It is acidic, and blue litmus turns red.",
        worked_solution="A pH lower than 7 indicates an acidic solution. Acids turn blue litmus red.",
        tags=["science", "chemistry", "ph"],
        estimated_time_sec=90,
    ),
    "c_common_salts_uses": _spec(
        difficulty="medium",
        question_type="chemistry",
        prompt="Name the salt used in baking powder and state one reason it is useful in cooking.",
        answer="Baking soda; it releases carbon dioxide on heating and helps dough rise.",
        worked_solution="Baking soda is sodium hydrogen carbonate. On heating or in acidic conditions it releases carbon dioxide, which makes dough or batter rise.",
        tags=["science", "chemistry", "salts"],
        estimated_time_sec=120,
    ),
    "c_reactivity_series": _spec(
        difficulty="medium",
        question_type="chemistry",
        prompt="Why does iron displace copper from copper sulphate solution, but copper does not displace iron from iron sulphate solution?",
        answer="Iron is more reactive than copper.",
        worked_solution="In the reactivity series, iron lies above copper, so iron can displace copper from its salt solution. Copper is less reactive, so it cannot displace iron.",
        tags=["science", "chemistry", "reactivity"],
        estimated_time_sec=150,
    ),
    "c_corrosion_prevention": _spec(
        difficulty="easy",
        question_type="chemistry",
        prompt="What is corrosion? Mention one method used to prevent rusting of iron.",
        answer="Corrosion is the slow deterioration of a metal by reaction with the environment; painting prevents rusting.",
        worked_solution="Corrosion is the gradual damage of metals caused by air, moisture, or chemicals. Rusting of iron can be prevented by painting, galvanising, oiling, or greasing.",
        tags=["science", "chemistry", "corrosion"],
        estimated_time_sec=90,
    ),
    "c_covalent_bonding_hydrocarbons": _spec(
        difficulty="medium",
        question_type="chemistry",
        prompt="Differentiate between saturated and unsaturated hydrocarbons with one example each.",
        answer="Saturated hydrocarbons have only single bonds, for example methane; unsaturated hydrocarbons have double or triple bonds, for example ethene.",
        worked_solution="Saturated hydrocarbons contain only single covalent bonds between carbon atoms, like methane. Unsaturated hydrocarbons contain double or triple bonds, like ethene or ethyne.",
        tags=["science", "chemistry", "hydrocarbons"],
        estimated_time_sec=120,
    ),
    "c_ethanol_soap_detergents": _spec(
        difficulty="medium",
        question_type="chemistry",
        prompt="Why does soap not work well with hard water?",
        answer="Soap forms insoluble scum with calcium and magnesium ions in hard water.",
        worked_solution="Hard water contains calcium and magnesium ions. Soap reacts with these ions to form insoluble precipitates called scum, so less soap is available for cleansing.",
        tags=["science", "chemistry", "soap"],
        estimated_time_sec=120,
    ),
    "c_nutrition_respiration": _spec(
        difficulty="easy",
        question_type="biology",
        prompt="Why is photosynthesis considered essential for life on Earth?",
        answer="It produces food and releases oxygen.",
        worked_solution="During photosynthesis, green plants make food from carbon dioxide and water using sunlight. This process supports food chains and releases oxygen needed by most living organisms.",
        tags=["science", "biology", "photosynthesis"],
        estimated_time_sec=120,
    ),
    "c_transport_excretion": _spec(
        difficulty="medium",
        question_type="biology",
        prompt="Name the plant tissue that transports water and minerals and the tissue that transports food.",
        answer="Xylem transports water and minerals; phloem transports food.",
        worked_solution="In plants, xylem carries water and dissolved minerals from the roots upward, while phloem transports prepared food to different parts of the plant.",
        tags=["science", "biology", "transport"],
        estimated_time_sec=90,
    ),
    "c_tropism_plant_hormones": _spec(
        difficulty="easy",
        question_type="biology",
        prompt="What is phototropism? Give one example.",
        answer="Phototropism is the growth of a plant part in response to light; a shoot bending towards light is an example.",
        worked_solution="Phototropism is a directional growth response caused by light. For example, the shoot of a potted plant bends towards sunlight.",
        tags=["science", "biology", "tropism"],
        estimated_time_sec=90,
    ),
    "c_nervous_system_hormones": _spec(
        difficulty="medium",
        question_type="biology",
        prompt="Differentiate between a reflex action and a voluntary action.",
        answer="A reflex action is automatic and immediate, while a voluntary action is under conscious control.",
        worked_solution="Reflex actions such as withdrawing a hand from a hot object happen quickly without conscious thinking. Voluntary actions such as writing are controlled consciously by the brain.",
        tags=["science", "biology", "reflex_action"],
        estimated_time_sec=120,
    ),
    "c_asexual_reproduction": _spec(
        difficulty="easy",
        question_type="biology",
        prompt="Name one organism that reproduces by budding and one that reproduces by binary fission.",
        answer="Yeast reproduces by budding and Amoeba reproduces by binary fission.",
        worked_solution="In budding, a small outgrowth develops into a new individual, as in yeast. In binary fission, one cell divides into two similar cells, as in Amoeba.",
        tags=["science", "biology", "reproduction"],
        estimated_time_sec=90,
    ),
    "c_human_reproductive_health": _spec(
        difficulty="medium",
        question_type="biology",
        prompt="Why is safe sex important in human reproductive health?",
        answer="It helps prevent sexually transmitted infections and unwanted pregnancies.",
        worked_solution="Safe sexual practices reduce the risk of HIV/AIDS and other sexually transmitted infections and help in responsible family planning.",
        tags=["science", "biology", "reproductive_health"],
        estimated_time_sec=120,
    ),
    "c_mendel_inheritance": _spec(
        difficulty="medium",
        question_type="biology",
        prompt="In pea plants, tallness is dominant over dwarfness. What phenotypic ratio is expected in the F2 generation of a monohybrid cross?",
        answer="3:1",
        worked_solution="A monohybrid cross between two heterozygous tall plants gives three tall offspring for every one dwarf offspring. So the F2 phenotypic ratio is 3:1.",
        tags=["science", "biology", "mendel"],
        estimated_time_sec=150,
    ),
    "c_sex_determination": _spec(
        difficulty="easy",
        question_type="biology",
        prompt="Which parent determines the sex of a child in humans and why?",
        answer="The father determines the sex because sperm may carry either X or Y chromosome.",
        worked_solution="The mother always contributes an X chromosome. The father contributes either X or Y through sperm, so the father determines whether the child is XX or XY.",
        tags=["science", "biology", "sex_determination"],
        estimated_time_sec=120,
    ),
    "c_spherical_mirrors": _spec(
        difficulty="medium",
        question_type="physics",
        prompt="Name the mirror commonly used as a rear-view mirror in vehicles and state one reason.",
        answer="Convex mirror, because it provides a wider field of view.",
        worked_solution="Convex mirrors form diminished, erect images and cover a larger area behind the vehicle, so they are used as rear-view mirrors.",
        tags=["science", "physics", "mirrors"],
        estimated_time_sec=90,
    ),
    "c_lenses_refraction": _spec(
        difficulty="medium",
        question_type="physics",
        prompt="A ray of light passes from air into glass. Does it bend towards the normal or away from the normal? Why?",
        answer="Towards the normal, because glass is optically denser than air.",
        worked_solution="When light enters a denser medium from a rarer medium, its speed decreases and it bends towards the normal. Glass is denser than air, so the ray bends towards the normal.",
        tags=["science", "physics", "refraction"],
        estimated_time_sec=120,
    ),
    "c_eye_defects_correction": _spec(
        difficulty="easy",
        question_type="physics",
        prompt="What type of lens is used to correct myopia?",
        answer="A concave lens.",
        worked_solution="In myopia, distant objects appear blurred because the image forms in front of the retina. A concave lens diverges light rays and shifts the image back onto the retina.",
        tags=["science", "physics", "human_eye"],
        estimated_time_sec=90,
    ),
    "c_dispersion_scattering": _spec(
        difficulty="medium",
        question_type="physics",
        prompt="Why is the sky blue during the day?",
        answer="Because shorter-wavelength blue light is scattered more by the atmosphere.",
        worked_solution="Atmospheric particles scatter shorter wavelengths of sunlight more strongly than longer wavelengths. Blue light is therefore scattered in all directions and reaches our eyes from all parts of the sky.",
        tags=["science", "physics", "scattering"],
        estimated_time_sec=120,
    ),
    "c_ohms_law_resistance": _spec(
        difficulty="medium",
        question_type="physics",
        prompt="A resistor has potential difference 6 V across it and current 2 A through it. Find its resistance.",
        answer="3 ohm",
        worked_solution="Using Ohm’s law, R = V/I = 6/2 = 3 ohm.",
        tags=["science", "physics", "ohms_law"],
        estimated_time_sec=90,
    ),
    "c_electric_power_heating": _spec(
        difficulty="medium",
        question_type="physics",
        prompt="An electric iron draws 5 A current from a 220 V supply. Find its power.",
        answer="1100 W",
        worked_solution="Electric power P = VI = 220 x 5 = 1100 W.",
        tags=["science", "physics", "electric_power"],
        estimated_time_sec=90,
    ),
    "c_magnetic_field_fleming": _spec(
        difficulty="medium",
        question_type="physics",
        prompt="State one use of Fleming’s Left Hand Rule.",
        answer="It is used to find the direction of force on a current-carrying conductor in a magnetic field.",
        worked_solution="Fleming’s Left Hand Rule gives the relative directions of magnetic field, current, and force. It helps predict the direction of motion in devices like electric motors.",
        tags=["science", "physics", "fleming_rule"],
        estimated_time_sec=120,
    ),
    "c_domestic_circuits_ac_dc": _spec(
        difficulty="easy",
        question_type="physics",
        prompt="Why are household appliances connected in parallel in domestic circuits?",
        answer="So each appliance gets the same voltage and can operate independently.",
        worked_solution="In a parallel connection, all appliances receive the same potential difference. Also, switching off one appliance does not interrupt the others.",
        tags=["science", "physics", "domestic_circuits"],
        estimated_time_sec=120,
    ),
    "c_ecosystem_food_chains": _spec(
        difficulty="easy",
        question_type="biology",
        prompt="What is meant by a food chain?",
        answer="A food chain is a sequence in which one organism is eaten by another, showing transfer of energy.",
        worked_solution="A food chain shows how energy passes from producers to consumers and then to higher consumers in an ecosystem.",
        tags=["science", "biology", "ecosystem"],
        estimated_time_sec=90,
    ),
    "c_biodegradable_ozone": _spec(
        difficulty="medium",
        question_type="biology",
        prompt="Differentiate between biodegradable and non-biodegradable substances with one example each.",
        answer="Biodegradable substances can be decomposed by microorganisms, for example vegetable peels; non-biodegradable substances do not decompose easily, for example plastic.",
        worked_solution="Biodegradable materials are broken down naturally by decomposers, while non-biodegradable materials persist in the environment for long periods. Vegetable waste is biodegradable, but plastic is not.",
        tags=["science", "biology", "biodegradable"],
        estimated_time_sec=120,
    ),
}
