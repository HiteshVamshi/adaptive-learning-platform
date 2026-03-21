from __future__ import annotations

from adaptive_learning.data.schemas import ConceptRecord, RelationshipRecord, SyllabusTopicRecord, TextbookSectionRecord


CBSE_SCIENCE_SOURCE_DOCUMENT = "CBSE Science Subject Code 086 Classes IX and X (2025-26)"
CBSE_SCIENCE_SOURCE_URL = "https://cbseacademic.nic.in/web_material/CurriculumMain26/Sec/Science_Sec_2025-26.pdf"
NCERT_SCIENCE_SOURCE_DOCUMENT = "NCERT Science Textbook for Class X (Contents)"
NCERT_SCIENCE_SOURCE_URL = "https://ncert.nic.in/textbook/pdf/jesc1ps.pdf"
NCERT_SCIENCE_INDEX_URL = "https://ncert.nic.in/textbook.php?jesc1=ps-13"
NCERT_SCIENCE_CHAPTER_PDF_TEMPLATE = "https://ncert.nic.in/textbook/pdf/jesc1{chapter_number:02d}.pdf"
CLASS_LEVEL = "CBSE_10"
SYLLABUS_SESSION = "2025-26"


SCIENCE_TOPIC_ROWS = [
    ("topic_chemical_reactions", "Chemical Reactions and Equations", "unit_chemistry", "Chemical Substances - Nature and Behaviour", 25, "ch_chemical_reactions", "Chemical Reactions and Equations", 0, "Chemical reactions, Chemical equation, Balanced chemical equation, types of chemical reactions: combination, decomposition, displacement, double displacement, precipitation, endothermic exothermic reactions, oxidation and reduction.", "L295-L299"),
    ("topic_acids_bases_salts", "Acids, Bases and Salts", "unit_chemistry", "Chemical Substances - Nature and Behaviour", 25, "ch_acids_bases_salts", "Acids, Bases and Salts", 0, "Acids and Bases – definitions in terms of furnishing of H+ and OH– ions, identification using indicators, chemical properties, examples and uses, neutralization, concept of pH scale, importance of pH in everyday life; preparation and uses of Sodium Hydroxide, Bleaching powder, Baking soda, Washing soda and Plaster of Paris.", "L300-L304"),
    ("topic_metals_non_metals", "Metals and Non-metals", "unit_chemistry", "Chemical Substances - Nature and Behaviour", 25, "ch_metals_non_metals", "Metals and Non-metals", 0, "Properties of metals and non-metals; Reactivity series; Formation and properties of ionic compounds; Basic metallurgical processes; Corrosion and its prevention.", "L305-L307"),
    ("topic_carbon_compounds", "Carbon and its Compounds", "unit_chemistry", "Chemical Substances - Nature and Behaviour", 25, "ch_carbon_compounds", "Carbon and its Compounds", 0, "Covalent bonds – formation and properties of covalent compounds, Versatile nature of carbon, Hydrocarbons – saturated and unsaturated, Homologous series, nomenclature, chemical properties, Ethanol and Ethanoic acid, soaps and detergents.", "L308-L313"),
    ("topic_life_processes", "Life Processes", "unit_world_living", "World of Living", 25, "ch_life_processes", "Life Processes", 0, "Basic concept of nutrition, respiration, transport and excretion in plants and animals.", "L314-L317"),
    ("topic_control_coordination", "Control and Coordination", "unit_world_living", "World of Living", 25, "ch_control_coordination", "Control and Coordination", 0, "Control and co-ordination in animals and plants: Tropic movements in plants; Introduction of plant hormones; Nervous system; Voluntary, involuntary and reflex action; Chemical co-ordination: animal hormones.", "L318-L320"),
    ("topic_reproduction", "Reproduction", "unit_world_living", "World of Living", 25, "ch_reproduction", "How do Organisms Reproduce?", 0, "Reproduction in animals and plants (asexual and sexual), reproductive health, need and methods of family planning, safe sex vs HIV/AIDS, child bearing and women’s health.", "L321-L323"),
    ("topic_heredity", "Heredity", "unit_world_living", "World of Living", 25, "ch_heredity", "Heredity", 0, "Heredity; Mendel’s contribution - Laws for inheritance of traits; Sex determination; brief introduction.", "L324-L325"),
    ("topic_light_reflection", "Light - Reflection and Refraction", "unit_natural_phenomena", "Natural Phenomena", 12, "ch_light_reflection_refraction", "Light - Reflection and Refraction", 0, "Reflection of light by curved surfaces; Images formed by spherical mirrors; Refraction; Laws of refraction; Refraction of light by spherical lens; Lens formula; Magnification; Power of a lens.", "L326-L333"),
    ("topic_human_eye", "The Human Eye and the Colourful World", "unit_natural_phenomena", "Natural Phenomena", 12, "ch_human_eye_colourful_world", "The Human Eye and the Colourful World", 0, "Functioning of a lens in human eye, defects of vision and their corrections, applications of spherical mirrors and lenses, refraction of light through a prism, dispersion of light, scattering of light, applications in daily life.", "L334-L337"),
    ("topic_electricity", "Electricity", "unit_effects_current", "Effects of Current", 13, "ch_electricity", "Electricity", 0, "Electric current, potential difference and electric current; Ohm’s law; Resistance, Resistivity, factors on which the resistance of a conductor depends; series and parallel combination of resistors; Heating effect of electric current; Electric power, interrelation between P, V, I and R.", "L338-L344"),
    ("topic_magnetic_effects", "Magnetic Effects of Electric Current", "unit_effects_current", "Effects of Current", 13, "ch_magnetic_effects_current", "Magnetic Effects of Electric Current", 0, "Magnetic field, field lines, field due to a current carrying conductor, field due to current carrying coil or solenoid; Force on current carrying conductor, Fleming’s Left Hand Rule, Direct current, Alternating current, Domestic electric circuits.", "L345-L348"),
    ("topic_environment", "Our Environment", "unit_natural_resources", "Natural Resources", 5, "ch_our_environment", "Our Environment", 0, "Eco-system, Environmental problems, Ozone depletion, waste production and their solutions. Biodegradable and non-biodegradable substances.", "L349-L352"),
]


SCIENCE_TEXTBOOK_ROWS = [
    (1, "ch_chemical_reactions", "Chemical Reactions and Equations", 1, "L273-L273"),
    (2, "ch_acids_bases_salts", "Acids, Bases and Salts", 17, "L274-L274"),
    (3, "ch_metals_non_metals", "Metals and Non-metals", 37, "L275-L275"),
    (4, "ch_carbon_compounds", "Carbon and its Compounds", 58, "L276-L276"),
    (5, "ch_life_processes", "Life Processes", 79, "L277-L277"),
    (6, "ch_control_coordination", "Control and Coordination", 100, "L278-L278"),
    (7, "ch_reproduction", "How do Organisms Reproduce?", 113, "L279-L279"),
    (8, "ch_heredity", "Heredity", 128, "L280-L280"),
    (9, "ch_light_reflection_refraction", "Light - Reflection and Refraction", 134, "L281-L281"),
    (10, "ch_human_eye_colourful_world", "The Human Eye and the Colourful World", 161, "L282-L282"),
    (11, "ch_electricity", "Electricity", 171, "L283-L283"),
    (12, "ch_magnetic_effects_current", "Magnetic Effects of Electric Current", 195, "L284-L284"),
    (13, "ch_our_environment", "Our Environment", 208, "L285-L285"),
]


def syllabus_topic_seed() -> list[SyllabusTopicRecord]:
    return [
        SyllabusTopicRecord(
            topic_id=topic_id,
            topic_name=topic_name,
            class_level=CLASS_LEVEL,
            syllabus_session=SYLLABUS_SESSION,
            unit_id=unit_id,
            unit_name=unit_name,
            unit_marks=unit_marks,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
            periods=periods,
            official_text=official_text,
            source_lines=source_lines,
            source_document=CBSE_SCIENCE_SOURCE_DOCUMENT,
            source_url=CBSE_SCIENCE_SOURCE_URL,
        )
        for topic_id, topic_name, unit_id, unit_name, unit_marks, chapter_id, chapter_name, periods, official_text, source_lines in SCIENCE_TOPIC_ROWS
    ]


def textbook_section_seed() -> list[TextbookSectionRecord]:
    return [
        TextbookSectionRecord(
            section_id=f"{chapter_id}::{chapter_number}",
            class_level=CLASS_LEVEL,
            chapter_number=chapter_number,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
            section_number=str(chapter_number),
            section_title=chapter_name,
            start_page=start_page,
            chapter_pdf_url=NCERT_SCIENCE_CHAPTER_PDF_TEMPLATE.format(chapter_number=chapter_number),
            source_document=NCERT_SCIENCE_SOURCE_DOCUMENT,
            source_url=NCERT_SCIENCE_SOURCE_URL,
            source_lines=source_lines,
        )
        for chapter_number, chapter_id, chapter_name, start_page, source_lines in SCIENCE_TEXTBOOK_ROWS
    ]


def chapter_and_concept_seed() -> list[ConceptRecord]:
    records: list[ConceptRecord] = []

    def chapter(chapter_id: str, chapter_name: str, unit_name: str, unit_marks: int, order_index: int, description: str, objective: str, tags: list[str]) -> None:
        records.append(
            ConceptRecord(
                concept_id=chapter_id,
                name=chapter_name,
                node_type="chapter",
                chapter_id=chapter_id,
                chapter_name=chapter_name,
                parent_concept_id=None,
                class_level=CLASS_LEVEL,
                description=description,
                learning_objective=objective,
                tags=tags + ["cbse_official"],
                difficulty_band="core",
                order_index=order_index,
                source_document=CBSE_SCIENCE_SOURCE_DOCUMENT,
                source_url=CBSE_SCIENCE_SOURCE_URL,
                source_kind="official_text",
                syllabus_session=SYLLABUS_SESSION,
                unit_name=unit_name,
                unit_marks=unit_marks,
                periods=0,
            )
        )

    def concept(concept_id: str, name: str, chapter_id: str, chapter_name: str, order_index: int, description: str, objective: str, tags: list[str], difficulty: str) -> None:
        records.append(
            ConceptRecord(
                concept_id=concept_id,
                name=name,
                node_type="concept",
                chapter_id=chapter_id,
                chapter_name=chapter_name,
                parent_concept_id=chapter_id,
                class_level=CLASS_LEVEL,
                description=description,
                learning_objective=objective,
                tags=tags,
                difficulty_band=difficulty,
                order_index=order_index,
                source_document=CBSE_SCIENCE_SOURCE_DOCUMENT,
                source_url=CBSE_SCIENCE_SOURCE_URL,
                source_kind="decomposed_from_official_chapter",
                syllabus_session=SYLLABUS_SESSION,
            )
        )

    chapter("ch_chemical_reactions", "Chemical Reactions and Equations", "Chemical Substances - Nature and Behaviour", 25, 1, "Chemical changes, equations, balancing, and reaction patterns.", "Represent and analyse common chemical reactions correctly.", ["chemistry", "reactions"])
    concept("c_balancing_chemical_equations", "Balancing Chemical Equations", "ch_chemical_reactions", "Chemical Reactions and Equations", 11, "Balance symbolic equations to conserve atoms and mass.", "Balance simple chemical equations and explain why balancing is required.", ["chemistry", "balancing", "equations"], "easy")
    concept("c_types_chemical_reactions", "Types of Chemical Reactions", "ch_chemical_reactions", "Chemical Reactions and Equations", 12, "Classify reactions as combination, decomposition, displacement, double displacement, redox, endothermic, or exothermic.", "Identify reaction types from chemical equations and observations.", ["chemistry", "reaction_types", "redox"], "medium")

    chapter("ch_acids_bases_salts", "Acids, Bases and Salts", "Chemical Substances - Nature and Behaviour", 25, 2, "Acids, bases, indicators, pH, neutralisation, and useful salts.", "Relate acidic and basic behaviour to pH, indicators, and everyday chemicals.", ["chemistry", "ph", "salts"])
    concept("c_acids_bases_ph", "Acids, Bases and pH", "ch_acids_bases_salts", "Acids, Bases and Salts", 21, "Use indicators and pH to classify solutions and explain their behaviour.", "Predict acidic or basic behaviour using indicators and pH scale ideas.", ["chemistry", "acids", "bases", "ph"], "easy")
    concept("c_common_salts_uses", "Common Salts and Their Uses", "ch_acids_bases_salts", "Acids, Bases and Salts", 22, "Connect sodium hydroxide, bleaching powder, baking soda, washing soda, and plaster of Paris to preparation and use.", "Explain how common salts are prepared and where they are used.", ["chemistry", "salts", "applications"], "medium")

    chapter("ch_metals_non_metals", "Metals and Non-metals", "Chemical Substances - Nature and Behaviour", 25, 3, "Properties of metals, non-metals, reactivity, ionic compounds, and corrosion.", "Use reactivity and material properties to explain chemical behaviour of metals and non-metals.", ["chemistry", "metals", "non_metals"])
    concept("c_reactivity_series", "Reactivity Series and Ionic Compounds", "ch_metals_non_metals", "Metals and Non-metals", 31, "Compare metal reactivity and explain ionic compound formation.", "Use the reactivity series to predict displacement and ionic bonding outcomes.", ["chemistry", "reactivity", "ionic_compounds"], "medium")
    concept("c_corrosion_prevention", "Corrosion and Prevention", "ch_metals_non_metals", "Metals and Non-metals", 32, "Explain corrosion, rusting, and common prevention methods.", "Describe why corrosion happens and how it is prevented.", ["chemistry", "corrosion", "rusting"], "easy")

    chapter("ch_carbon_compounds", "Carbon and its Compounds", "Chemical Substances - Nature and Behaviour", 25, 4, "Covalent bonding, hydrocarbons, functional groups, ethanol, ethanoic acid, soaps, and detergents.", "Reason about carbon bonding, carbon compound properties, and everyday carbon chemistry.", ["chemistry", "carbon"])
    concept("c_covalent_bonding_hydrocarbons", "Covalent Bonding and Hydrocarbons", "ch_carbon_compounds", "Carbon and its Compounds", 41, "Model covalent bonding and classify hydrocarbons as saturated or unsaturated.", "Explain covalent bonds and identify basic hydrocarbon families.", ["chemistry", "carbon", "hydrocarbons"], "medium")
    concept("c_ethanol_soap_detergents", "Ethanol, Ethanoic Acid, Soaps and Detergents", "ch_carbon_compounds", "Carbon and its Compounds", 42, "Connect functional groups to ethanol, ethanoic acid, cleansing action, and soap behaviour.", "Relate common carbon compounds to properties, reactions, and cleansing use.", ["chemistry", "ethanol", "soap"], "medium")

    chapter("ch_life_processes", "Life Processes", "World of Living", 25, 5, "Nutrition, respiration, transport, and excretion in plants and animals.", "Compare how living organisms perform essential life processes.", ["biology", "life_processes"])
    concept("c_nutrition_respiration", "Nutrition and Respiration", "ch_life_processes", "Life Processes", 51, "Compare photosynthesis, nutrition, and respiration across organisms.", "Explain how organisms obtain and use energy through nutrition and respiration.", ["biology", "nutrition", "respiration"], "easy")
    concept("c_transport_excretion", "Transport and Excretion", "ch_life_processes", "Life Processes", 52, "Describe transport and excretion in plants and animals.", "Compare transport and waste removal systems in plants and animals.", ["biology", "transport", "excretion"], "medium")

    chapter("ch_control_coordination", "Control and Coordination", "World of Living", 25, 6, "Plant tropisms, nervous control, reflexes, and animal hormones.", "Explain how plants and animals sense, respond, and coordinate actions.", ["biology", "coordination"])
    concept("c_tropism_plant_hormones", "Tropisms and Plant Hormones", "ch_control_coordination", "Control and Coordination", 61, "Explain plant movements and hormone-mediated responses.", "Relate plant responses to tropisms and hormone action.", ["biology", "tropism", "plant_hormones"], "easy")
    concept("c_nervous_system_hormones", "Nervous System and Animal Hormones", "ch_control_coordination", "Control and Coordination", 62, "Connect neurons, reflexes, and endocrine control in animals.", "Explain reflex action, nervous coordination, and hormonal control.", ["biology", "nervous_system", "hormones"], "medium")

    chapter("ch_reproduction", "How do Organisms Reproduce?", "World of Living", 25, 7, "Asexual and sexual reproduction, reproductive health, and safe practices.", "Explain reproductive processes in organisms and connect them to health choices.", ["biology", "reproduction"])
    concept("c_asexual_reproduction", "Asexual and Sexual Reproduction", "ch_reproduction", "How do Organisms Reproduce?", 71, "Compare asexual and sexual reproduction in plants and animals.", "Differentiate reproduction modes with examples and outcomes.", ["biology", "asexual", "sexual_reproduction"], "easy")
    concept("c_human_reproductive_health", "Human Reproductive Health", "ch_reproduction", "How do Organisms Reproduce?", 72, "Relate human reproduction to reproductive health, family planning, and disease prevention.", "Explain reproductive health concepts and safe-sex reasoning.", ["biology", "reproductive_health", "family_planning"], "medium")

    chapter("ch_heredity", "Heredity", "World of Living", 25, 8, "Inheritance of traits and sex determination.", "Use heredity models to explain inheritance patterns and sex determination.", ["biology", "heredity"])
    concept("c_mendel_inheritance", "Mendelian Inheritance", "ch_heredity", "Heredity", 81, "Use Mendel’s laws to predict trait inheritance in simple crosses.", "Apply simple inheritance patterns to predict offspring traits.", ["biology", "mendel", "inheritance"], "medium")
    concept("c_sex_determination", "Sex Determination", "ch_heredity", "Heredity", 82, "Explain chromosomal sex determination in humans.", "Describe how sex is determined chromosomally in human beings.", ["biology", "sex_determination", "chromosomes"], "easy")

    chapter("ch_light_reflection_refraction", "Light - Reflection and Refraction", "Natural Phenomena", 12, 9, "Reflection, mirrors, refraction, lenses, and image formation.", "Use ray ideas and formulae to reason about mirrors and lenses.", ["physics", "light"])
    concept("c_spherical_mirrors", "Spherical Mirrors", "ch_light_reflection_refraction", "Light - Reflection and Refraction", 91, "Relate mirror geometry to image formation and mirror formula.", "Predict image characteristics for concave and convex mirrors.", ["physics", "mirrors", "reflection"], "medium")
    concept("c_lenses_refraction", "Refraction and Spherical Lenses", "ch_light_reflection_refraction", "Light - Reflection and Refraction", 92, "Apply laws of refraction and lens formula ideas to image formation.", "Explain refraction and lens-based image formation.", ["physics", "refraction", "lenses"], "medium")

    chapter("ch_human_eye_colourful_world", "The Human Eye and the Colourful World", "Natural Phenomena", 12, 10, "Eye function, defects of vision, prism, dispersion, and scattering.", "Connect optical principles to vision, colour phenomena, and atmosphere.", ["physics", "optics"])
    concept("c_eye_defects_correction", "Eye Defects and Their Correction", "ch_human_eye_colourful_world", "The Human Eye and the Colourful World", 101, "Relate myopia, hypermetropia, and correction lenses to image formation.", "Explain common vision defects and their correction using lenses.", ["physics", "human_eye", "vision"], "easy")
    concept("c_dispersion_scattering", "Dispersion and Scattering of Light", "ch_human_eye_colourful_world", "The Human Eye and the Colourful World", 102, "Explain prism dispersion, atmospheric scattering, and colour phenomena.", "Use dispersion and scattering to explain familiar optical effects.", ["physics", "dispersion", "scattering"], "medium")

    chapter("ch_electricity", "Electricity", "Effects of Current", 13, 11, "Current, potential difference, resistance, circuits, power, and heating.", "Apply circuit laws and power relations to everyday electrical situations.", ["physics", "electricity"])
    concept("c_ohms_law_resistance", "Ohm's Law and Resistance", "ch_electricity", "Electricity", 111, "Relate current, voltage, resistance, and resistivity.", "Apply Ohm’s law and resistance relationships in circuit problems.", ["physics", "ohms_law", "resistance"], "medium")
    concept("c_electric_power_heating", "Electric Power and Heating Effect", "ch_electricity", "Electricity", 112, "Connect power, energy use, and heating effect of current.", "Compute power or energy use and explain heating devices.", ["physics", "power", "heating_effect"], "medium")

    chapter("ch_magnetic_effects_current", "Magnetic Effects of Electric Current", "Effects of Current", 13, 12, "Magnetic field, force on conductors, Fleming’s rule, AC/DC, and domestic circuits.", "Use field-line and motor-effect ideas to explain current and magnetism.", ["physics", "magnetism"])
    concept("c_magnetic_field_fleming", "Magnetic Field and Fleming's Left Hand Rule", "ch_magnetic_effects_current", "Magnetic Effects of Electric Current", 121, "Relate current-carrying conductors to magnetic fields and force direction.", "Explain magnetic field patterns and force direction using Fleming’s rule.", ["physics", "magnetic_field", "fleming_rule"], "medium")
    concept("c_domestic_circuits_ac_dc", "AC, DC and Domestic Circuits", "ch_magnetic_effects_current", "Magnetic Effects of Electric Current", 122, "Compare AC and DC and reason about safe domestic wiring.", "Explain domestic circuits, safety, and why AC is preferred in transmission.", ["physics", "ac_dc", "domestic_circuits"], "easy")

    chapter("ch_our_environment", "Our Environment", "Natural Resources", 5, 13, "Ecosystems, waste, ozone depletion, and biodegradable versus non-biodegradable substances.", "Explain ecosystem structure and common environmental challenges.", ["biology", "environment"])
    concept("c_ecosystem_food_chains", "Ecosystems and Food Chains", "ch_our_environment", "Our Environment", 131, "Relate organisms, trophic levels, and food chains inside ecosystems.", "Explain ecosystem components and food-chain structure.", ["biology", "ecosystem", "food_chain"], "easy")
    concept("c_biodegradable_ozone", "Biodegradable Waste and Ozone Layer", "ch_our_environment", "Our Environment", 132, "Differentiate biodegradable and non-biodegradable waste and connect human action to ozone depletion.", "Explain waste classification and ozone-related environmental issues.", ["biology", "biodegradable", "ozone"], "medium")

    return records


def relationship_seed() -> list[RelationshipRecord]:
    rows = [
        ("c_balancing_chemical_equations", "c_types_chemical_reactions", "prerequisite", 1.0, "Balanced equations support reasoning about reaction classes."),
        ("c_acids_bases_ph", "c_common_salts_uses", "prerequisite", 0.8, "Understanding acid-base behaviour helps explain common salts and their uses."),
        ("c_reactivity_series", "c_corrosion_prevention", "prerequisite", 0.8, "Reactivity concepts support explanations of corrosion and prevention."),
        ("c_covalent_bonding_hydrocarbons", "c_ethanol_soap_detergents", "prerequisite", 0.8, "Bonding and hydrocarbon ideas support later carbon-compound applications."),
        ("c_nutrition_respiration", "c_transport_excretion", "related", 0.7, "Life-process topics are tightly connected in plants and animals."),
        ("c_tropism_plant_hormones", "c_nervous_system_hormones", "related", 0.7, "Both describe organism control and coordination mechanisms."),
        ("c_asexual_reproduction", "c_human_reproductive_health", "prerequisite", 0.7, "General reproduction ideas support human reproductive-health reasoning."),
        ("c_mendel_inheritance", "c_sex_determination", "prerequisite", 0.8, "Trait inheritance is foundational before specific sex-determination models."),
        ("c_spherical_mirrors", "c_lenses_refraction", "related", 0.6, "Both concern image formation through optical devices."),
        ("c_lenses_refraction", "c_eye_defects_correction", "prerequisite", 0.9, "Vision-defect correction depends on lens behaviour."),
        ("c_eye_defects_correction", "c_dispersion_scattering", "related", 0.5, "Both are optical applications within the same chapter."),
        ("c_ohms_law_resistance", "c_electric_power_heating", "prerequisite", 0.9, "Power calculations depend on current-voltage-resistance relations."),
        ("c_ohms_law_resistance", "c_magnetic_field_fleming", "related", 0.5, "Both build from current-carrying conductor ideas."),
        ("c_magnetic_field_fleming", "c_domestic_circuits_ac_dc", "prerequisite", 0.6, "Field and current ideas support circuit-system reasoning."),
        ("c_ecosystem_food_chains", "c_biodegradable_ozone", "related", 0.7, "Environmental structure and environmental problems are conceptually linked."),
    ]
    return [
        RelationshipRecord(
            source_concept_id=source,
            target_concept_id=target,
            relation_type=relation_type,
            weight=weight,
            rationale=rationale,
        )
        for source, target, relation_type, weight, rationale in rows
    ]
