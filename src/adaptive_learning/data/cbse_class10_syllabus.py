from __future__ import annotations

from adaptive_learning.data.schemas import ConceptRecord, RelationshipRecord, SyllabusTopicRecord


CBSE_MATH_SOURCE_DOCUMENT = "CBSE Mathematics (IX-X) Code No. 041 Session 2024-25"
CBSE_MATH_SOURCE_URL = (
    "https://cbseacademic.nic.in/web_material/CurriculumMain25/Sec/Maths_Sec_2024-25.pdf"
)
CBSE_MATH_SESSION = "2024-25"
CBSE_CLASS_LEVEL = "CBSE_10"

TOPIC_ROWS = [
    ("topic_real_numbers", "Real Numbers", "unit_number_systems", "Number Systems", 6, "ch_real_numbers", "Real Numbers", 15, "Fundamental Theorem of Arithmetic, irrationality, and decimal expansions of rational numbers.", "L189-L193", "Chapter scope from the official CBSE Class X syllabus."),
    ("topic_polynomials", "Polynomials", "unit_algebra", "Algebra", 20, "ch_polynomials", "Polynomials", 8, "Zeros of a polynomial and relationship between zeroes and coefficients of quadratic polynomials.", "L193-L196", ""),
    ("topic_pair_linear_equations", "Pair of Linear Equations in Two Variables", "unit_algebra", "Algebra", 20, "ch_pair_linear_equations", "Pair of Linear Equations in Two Variables", 15, "Graphical solution, consistency or inconsistency, substitution, elimination, and simple situational problems.", "L197-L201", ""),
    ("topic_quadratic_equations", "Quadratic Equations", "unit_algebra", "Algebra", 20, "ch_quadratic_equations", "Quadratic Equations", 15, "Standard form, factorization, quadratic formula, discriminant, and situational problems.", "L202-L207", ""),
    ("topic_arithmetic_progressions", "Arithmetic Progressions", "unit_algebra", "Algebra", 20, "ch_arithmetic_progressions", "Arithmetic Progressions", 10, "Nth term, sum of first n terms, and daily-life applications.", "L208-L210", ""),
    ("topic_coordinate_geometry", "Coordinate Geometry", "unit_coordinate_geometry", "Coordinate Geometry", 6, "ch_coordinate_geometry", "Coordinate Geometry", 15, "Review of graphs of linear equations, distance formula, and internal section formula.", "L211-L214", ""),
    ("topic_triangles", "Triangles", "unit_geometry", "Geometry", 15, "ch_triangles", "Triangles", 15, "Similar triangles, Basic Proportionality Theorem and converse, and similarity criteria.", "L215-L227", ""),
    ("topic_circles", "Circles", "unit_geometry", "Geometry", 15, "ch_circles", "Circles", 10, "Tangent at point of contact, perpendicular radius theorem, and equal tangents from an external point.", "L228-L232", ""),
    ("topic_intro_trigonometry", "Introduction to Trigonometry", "unit_trigonometry", "Trigonometry", 12, "ch_trigonometry", "Introduction to Trigonometry", 10, "Acute-angle trigonometric ratios, standard values, and relationships between ratios.", "L233-L237", ""),
    ("topic_trig_identities", "Trigonometric Identities", "unit_trigonometry", "Trigonometry", 12, "ch_trigonometry", "Introduction to Trigonometry", 15, "Proof and applications of sin^2 A + cos^2 A = 1, with simple identities.", "L238-L240", ""),
    ("topic_heights_distances", "Heights and Distances", "unit_trigonometry", "Trigonometry", 12, "ch_trigonometry", "Introduction to Trigonometry", 10, "Angles of elevation and depression with simple right-triangle applications.", "L241-L243", ""),
    ("topic_areas_related_circles", "Areas Related to Circles", "unit_mensuration", "Mensuration", 10, "ch_areas_related_to_circles", "Areas Related to Circles", 12, "Area of sectors and segments, plus perimeter and area problems on related plane figures.", "L244-L248", ""),
    ("topic_surface_areas_volumes", "Surface Areas and Volumes", "unit_mensuration", "Mensuration", 10, "ch_surface_areas_volumes", "Surface Areas and Volumes", 12, "Surface area and volume of combinations of two standard solids.", "L249-L251", ""),
    ("topic_statistics", "Statistics", "unit_statistics_probability", "Statistics and Probability", 11, "ch_statistics", "Statistics", 18, "Mean, median and mode of grouped data.", "L252-L254", ""),
    ("topic_probability", "Probability", "unit_statistics_probability", "Statistics and Probability", 11, "ch_probability", "Probability", 10, "Classical definition of probability and simple event problems.", "L255-L257", ""),
]

CHAPTER_ROWS = [
    ("ch_real_numbers", "Real Numbers", "Number Systems", 6, 15, 1, "Official CBSE Class X chapter on number-theoretic structure, irrationality, and decimal expansion.", "Use arithmetic structure and proof-based reasoning to analyze real numbers.", ["number_systems", "cbse_official"]),
    ("ch_polynomials", "Polynomials", "Algebra", 20, 8, 2, "Official CBSE Class X chapter on zeroes of polynomials and coefficient relationships.", "Analyze polynomial roots and their connection to coefficients.", ["algebra", "cbse_official"]),
    ("ch_pair_linear_equations", "Pair of Linear Equations in Two Variables", "Algebra", 20, 15, 3, "Official CBSE Class X chapter on solving and analyzing two-variable linear systems.", "Solve pairs of linear equations graphically and algebraically.", ["algebra", "systems", "cbse_official"]),
    ("ch_quadratic_equations", "Quadratic Equations", "Algebra", 20, 15, 4, "Official CBSE Class X chapter on solving quadratic equations and interpreting roots.", "Solve quadratic equations and explain how the discriminant controls root behavior.", ["algebra", "quadratics", "cbse_official"]),
    ("ch_arithmetic_progressions", "Arithmetic Progressions", "Algebra", 20, 10, 5, "Official CBSE Class X chapter on nth term, sum, and applications of arithmetic progressions.", "Reason about linear sequences and their sums in symbolic and applied settings.", ["sequences", "cbse_official"]),
    ("ch_coordinate_geometry", "Coordinate Geometry", "Coordinate Geometry", 6, 15, 6, "Official CBSE Class X chapter on distance and internal section formula on the Cartesian plane.", "Use coordinate methods to compute geometric quantities algebraically.", ["coordinates", "geometry", "cbse_official"]),
    ("ch_triangles", "Triangles", "Geometry", 15, 15, 7, "Official CBSE Class X chapter on similarity and proportionality in triangles.", "Use similarity and proportionality theorems to solve geometric problems.", ["geometry", "similarity", "cbse_official"]),
    ("ch_circles", "Circles", "Geometry", 15, 10, 8, "Official CBSE Class X chapter on tangents to a circle and related theorems.", "Use tangent properties for proof and length-reasoning problems.", ["circles", "geometry", "cbse_official"]),
    ("ch_trigonometry", "Introduction to Trigonometry", "Trigonometry", 12, 10, 9, "Official CBSE Class X trigonometry chapter on acute-angle ratios, identities, and applications.", "Represent right-triangle geometry numerically through trigonometric ratios and identities.", ["trigonometry", "cbse_official"]),
    ("ch_areas_related_to_circles", "Areas Related to Circles", "Mensuration", 10, 12, 10, "Official CBSE Class X chapter on sectors, segments, and related perimeter-area problems.", "Compute area and perimeter of sectors, segments, and circle-based plane figures.", ["mensuration", "circles", "cbse_official"]),
    ("ch_surface_areas_volumes", "Surface Areas and Volumes", "Mensuration", 10, 12, 11, "Official CBSE Class X chapter on combinations of solid figures.", "Compute surface area and volume for solids built from two standard 3D shapes.", ["mensuration", "solids", "cbse_official"]),
    ("ch_statistics", "Statistics", "Statistics and Probability", 11, 18, 12, "Official CBSE Class X chapter on mean, median and mode of grouped data.", "Summarize grouped data using central tendency measures.", ["statistics", "grouped_data", "cbse_official"]),
    ("ch_probability", "Probability", "Statistics and Probability", 11, 10, 13, "Official CBSE Class X chapter on the classical definition of probability.", "Compute simple event probabilities using the classical framework.", ["probability", "cbse_official"]),
]

CONCEPT_ROWS = [
    ("c_fundamental_theorem_arithmetic", "Fundamental Theorem of Arithmetic", "ch_real_numbers", "Real Numbers", "Use unique prime factorization to reason about divisibility, HCF, LCM, and denominator structure.", "Apply prime factorization as the central theorem for Class X real-number reasoning.", ["prime_factorization", "divisibility", "theorem"], "medium", 11, "official_text"),
    ("c_euclid_division_lemma", "Euclid Division Lemma", "ch_real_numbers", "Real Numbers", "Represent an integer as a = bq + r and use that decomposition to support HCF reasoning.", "Use Euclid-style quotient and remainder decomposition as a bridge into HCF computations.", ["division", "algorithm", "review"], "easy", 12, "decomposed_from_official_chapter"),
    ("c_hcf_lcm", "HCF and LCM", "ch_real_numbers", "Real Numbers", "Connect prime factorization and algorithmic reasoning to highest common factor and least common multiple.", "Compute HCF and LCM using theorem-based number reasoning.", ["hcf", "lcm", "prime_factorization"], "medium", 13, "decomposed_from_official_chapter"),
    ("c_irrational_numbers", "Irrational Numbers", "ch_real_numbers", "Real Numbers", "Use contradiction-based arguments to show that selected numbers are irrational.", "Distinguish rational and irrational numbers with proof-oriented reasoning.", ["irrationality", "proof", "number_system"], "hard", 14, "official_text"),
    ("c_decimal_expansions", "Decimal Expansions of Rational Numbers", "ch_real_numbers", "Real Numbers", "Classify rational numbers by terminating or recurring decimal expansion through denominator factorization.", "Explain decimal behavior of rational numbers from prime factors of the denominator.", ["decimals", "rational_numbers", "classification"], "medium", 15, "official_text"),
    ("c_polynomial_degree_terms", "Polynomial Degree and Terms", "ch_polynomials", "Polynomials", "Review polynomial vocabulary including terms, coefficients, and degree as scaffolding for root analysis.", "Read polynomial structure before applying Class X zero-based reasoning.", ["polynomial_structure", "review"], "easy", 21, "bridge_prerequisite"),
    ("c_zeroes_of_polynomial", "Zeroes of a Polynomial", "ch_polynomials", "Polynomials", "Find and interpret the zeroes of a polynomial in line with the official Class X syllabus.", "Compute and reason about polynomial zeroes from algebraic forms.", ["zeroes", "roots", "algebra"], "medium", 22, "official_text"),
    ("c_zero_coeff_relationship", "Relationship Between Zeroes and Coefficients", "ch_polynomials", "Polynomials", "Use the sum and product of zeroes to connect quadratic roots with coefficients.", "Construct or analyze quadratic polynomials using root-coefficient relationships.", ["quadratic", "coefficients", "roots"], "medium", 23, "official_text"),
    ("c_factorisation_remainder", "Factorisation and Remainder Theorem", "ch_polynomials", "Polynomials", "Use polynomial evaluation and factor checks as a supporting skill for zero-based reasoning.", "Test factors efficiently while linking factorization to roots.", ["factorisation", "evaluation", "bridge_skill"], "hard", 24, "bridge_prerequisite"),
    ("c_pair_linear_equations_graphical", "Graphical Solution of Pair of Linear Equations", "ch_pair_linear_equations", "Pair of Linear Equations in Two Variables", "Represent two linear equations on the Cartesian plane and interpret their point of intersection.", "Use graphs to solve linear systems and interpret geometric meaning.", ["graphs", "intersection", "linear_systems"], "easy", 31, "official_text"),
    ("c_pair_linear_equations_consistency", "Consistency and Number of Solutions", "ch_pair_linear_equations", "Pair of Linear Equations in Two Variables", "Use coefficient conditions and graph interpretation to determine whether a system is consistent, inconsistent, or dependent.", "Classify a pair of equations by its number of solutions.", ["consistency", "solution_count", "coefficients"], "medium", 32, "official_text"),
    ("c_pair_linear_equations_algebraic", "Algebraic Methods for Pair of Linear Equations", "ch_pair_linear_equations", "Pair of Linear Equations in Two Variables", "Solve linear systems by substitution or elimination and apply them in simple situational problems.", "Solve two-variable linear systems algebraically and in context.", ["substitution", "elimination", "applications"], "medium", 33, "official_text"),
    ("c_quadratic_factorization", "Quadratic Equations by Factorization", "ch_quadratic_equations", "Quadratic Equations", "Solve quadratic equations with real roots by expressing the polynomial as a product of linear factors.", "Solve suitable quadratic equations by factorization.", ["factorization", "real_roots", "quadratics"], "easy", 41, "official_text"),
    ("c_quadratic_formula", "Quadratic Formula", "ch_quadratic_equations", "Quadratic Equations", "Use the quadratic formula to solve any quadratic equation with real roots in standard form.", "Apply the quadratic formula accurately to Class X equations.", ["formula", "standard_form", "quadratics"], "medium", 42, "official_text"),
    ("c_discriminant_nature_roots", "Discriminant and Nature of Roots", "ch_quadratic_equations", "Quadratic Equations", "Use the discriminant to infer whether a quadratic equation has distinct, repeated, or no real roots.", "Relate discriminant value to the nature of quadratic roots.", ["discriminant", "root_nature", "quadratics"], "medium", 43, "official_text"),
    ("c_quadratic_applications", "Situational Problems on Quadratic Equations", "ch_quadratic_equations", "Quadratic Equations", "Translate day-to-day contexts into quadratic equations and solve them with an appropriate method.", "Model simple real-life situations using quadratic equations.", ["word_problems", "modelling", "applications"], "hard", 44, "official_text"),
    ("c_ap_nth_term", "Nth Term of an Arithmetic Progression", "ch_arithmetic_progressions", "Arithmetic Progressions", "Use the common difference and first term to derive or compute the nth term of an A.P.", "Find any requested term in an arithmetic progression.", ["ap", "nth_term", "sequence"], "easy", 51, "official_text"),
    ("c_ap_sum", "Sum of First n Terms of an Arithmetic Progression", "ch_arithmetic_progressions", "Arithmetic Progressions", "Derive and use the sum formula for the first n terms of an arithmetic progression.", "Compute cumulative totals in an arithmetic progression.", ["ap", "sum", "series"], "medium", 52, "official_text"),
    ("c_ap_applications", "Applications of Arithmetic Progressions", "ch_arithmetic_progressions", "Arithmetic Progressions", "Model and solve daily-life problems whose structure follows an arithmetic progression.", "Translate contextual patterns into A.P. terms or sums.", ["ap", "applications", "modelling"], "medium", 53, "official_text"),
    ("c_distance_formula", "Distance Formula", "ch_coordinate_geometry", "Coordinate Geometry", "Compute the distance between two points in the Cartesian plane using the standard formula.", "Apply the distance formula accurately in coordinate problems.", ["distance", "coordinates"], "easy", 61, "official_text"),
    ("c_section_formula", "Section Formula", "ch_coordinate_geometry", "Coordinate Geometry", "Find the point that divides a line segment internally in a given ratio.", "Use weighted averages to locate an internal division point.", ["section_formula", "ratio", "coordinates"], "medium", 62, "official_text"),
    ("c_area_of_triangle", "Area of a Triangle in Coordinates", "ch_coordinate_geometry", "Coordinate Geometry", "Use coordinate area reasoning to compute triangle area or check collinearity as an applied extension.", "Use coordinate-based area reasoning in triangle problems.", ["triangle_area", "coordinates", "collinearity"], "medium", 63, "decomposed_from_official_chapter"),
    ("c_similar_triangles", "Similar Triangles", "ch_triangles", "Triangles", "Identify and justify similarity between triangles from equal angles or proportional sides.", "Apply the Class X similarity criteria to establish triangle similarity.", ["similarity", "criteria", "proof"], "easy", 71, "official_text"),
    ("c_basic_proportionality_theorem", "Basic Proportionality Theorem", "ch_triangles", "Triangles", "If a line is drawn parallel to one side of a triangle, it divides the other two sides in the same ratio.", "Use BPT and its converse to solve segment-length problems in triangles.", ["bpt", "parallel_lines", "proportion"], "medium", 72, "official_text"),
    ("c_pythagoras_theorem", "Pythagoras Theorem", "ch_triangles", "Triangles", "Use right-triangle side relationships as a supporting skill for geometry and trigonometry problems.", "Compute unknown side lengths in right triangles.", ["pythagoras", "right_triangle", "bridge_skill"], "medium", 73, "bridge_prerequisite"),
    ("c_similarity_area_ratio", "Criteria for Similarity and Area Ratio", "ch_triangles", "Triangles", "Extend similarity reasoning to area ratios and proof-oriented questions on corresponding parts.", "Connect side-ratio similarity to area scaling arguments.", ["similarity", "area_ratio", "proof"], "hard", 74, "decomposed_from_official_chapter"),
    ("c_tangent_radius_perpendicular", "Tangent Perpendicular to Radius", "ch_circles", "Circles", "Use the theorem that the tangent at a point of a circle is perpendicular to the radius through that point.", "Apply the tangent-radius perpendicular theorem in proofs.", ["tangent", "radius", "proof"], "easy", 81, "official_text"),
    ("c_tangents_from_external_point", "Tangents from an External Point", "ch_circles", "Circles", "Use the theorem that tangents drawn from an external point to a circle are equal in length.", "Solve equal-tangent problems from an external point.", ["tangent_lengths", "circles", "theorem"], "medium", 82, "official_text"),
    ("c_trigonometric_ratios", "Trigonometric Ratios", "ch_trigonometry", "Introduction to Trigonometry", "Define sine, cosine, tangent and related ratios for an acute angle in a right-angled triangle.", "Compute and interpret trigonometric ratios from triangle sides.", ["sin", "cos", "tan"], "easy", 91, "official_text"),
    ("c_standard_trig_values", "Standard Trigonometric Values", "ch_trigonometry", "Introduction to Trigonometry", "Recall the exact trigonometric values at 30, 45 and 60 degrees within the official syllabus.", "Use standard angle values quickly in evaluation and simplification problems.", ["standard_values", "acute_angles", "trigonometry"], "easy", 92, "official_text"),
    ("c_trig_identities", "Trigonometric Identities", "ch_trigonometry", "Introduction to Trigonometry", "Use sin^2 A + cos^2 A = 1 and simple derived identities in proof and simplification tasks.", "Verify and apply simple Class X trigonometric identities.", ["identities", "proof", "simplification"], "medium", 93, "official_text"),
    ("c_heights_distances", "Heights and Distances", "ch_trigonometry", "Introduction to Trigonometry", "Solve simple elevation and depression problems using one or two right triangles and standard angles.", "Model height-and-distance situations with trigonometric ratios.", ["elevation", "depression", "applications"], "hard", 94, "official_text"),
    ("c_sectors_segments", "Sectors and Segments of a Circle", "ch_areas_related_to_circles", "Areas Related to Circles", "Compute areas of sectors and segments, with Class X restrictions on common central angles for segment problems.", "Calculate sector and segment areas in circle-based plane figures.", ["sector", "segment", "area"], "medium", 101, "official_text"),
    ("c_circle_perimeter_area_applications", "Perimeter and Area Applications on Circle Figures", "ch_areas_related_to_circles", "Areas Related to Circles", "Solve perimeter or circumference and area problems on figures built from arcs, sectors, and segments.", "Handle combined area-perimeter reasoning in circle-based figures.", ["circumference", "perimeter", "applications"], "hard", 102, "official_text"),
    ("c_combination_solids_surface_area", "Surface Area of Combined Solids", "ch_surface_areas_volumes", "Surface Areas and Volumes", "Compute external surface area when two standard solids are combined into one object.", "Set up and evaluate surface-area expressions for composite solids.", ["surface_area", "composite_solids", "mensuration"], "medium", 111, "official_text"),
    ("c_combination_solids_volume", "Volume of Combined Solids", "ch_surface_areas_volumes", "Surface Areas and Volumes", "Compute volume for composite solids formed from two standard Class X solid figures.", "Calculate volume of combined solids accurately.", ["volume", "composite_solids", "mensuration"], "medium", 112, "official_text"),
    ("c_grouped_data_mean", "Mean of Grouped Data", "ch_statistics", "Statistics", "Compute the mean of grouped data using tabular methods appropriate for Class X.", "Find the average of grouped data correctly.", ["mean", "grouped_data", "statistics"], "easy", 121, "official_text"),
    ("c_grouped_data_median_mode", "Median and Mode of Grouped Data", "ch_statistics", "Statistics", "Use the grouped-data formulas for median and mode, excluding bimodal cases as specified by CBSE.", "Compute median and mode of grouped data and interpret them.", ["median", "mode", "grouped_data"], "medium", 122, "official_text"),
    ("c_classical_probability", "Classical Probability", "ch_probability", "Probability", "Use favorable outcomes over total equally likely outcomes to compute simple probabilities.", "Solve introductory probability problems using the classical definition.", ["equally_likely", "event", "probability"], "easy", 131, "official_text"),
]

RELATIONSHIP_ROWS = [
    ("c_fundamental_theorem_arithmetic", "c_hcf_lcm", "prerequisite", 1.0, "Prime factorization reasoning supports HCF and LCM computation."),
    ("c_fundamental_theorem_arithmetic", "c_decimal_expansions", "prerequisite", 0.95, "Denominator prime factors determine whether a rational decimal terminates."),
    ("c_euclid_division_lemma", "c_hcf_lcm", "prerequisite", 0.9, "Division-lemma reasoning supports algorithmic HCF computations."),
    ("c_irrational_numbers", "c_decimal_expansions", "related", 0.55, "Both concepts distinguish rational and irrational number behavior."),
    ("c_polynomial_degree_terms", "c_zeroes_of_polynomial", "prerequisite", 0.8, "Structural understanding supports zero-finding."),
    ("c_zeroes_of_polynomial", "c_zero_coeff_relationship", "prerequisite", 1.0, "Root understanding precedes coefficient relationships."),
    ("c_zeroes_of_polynomial", "c_factorisation_remainder", "related", 0.75, "Factor checks and zeroes are tightly connected."),
    ("c_pair_linear_equations_graphical", "c_pair_linear_equations_consistency", "prerequisite", 0.95, "Graphical interpretation motivates the number of solutions."),
    ("c_pair_linear_equations_consistency", "c_pair_linear_equations_algebraic", "prerequisite", 0.85, "Understanding solution cases supports choosing algebraic methods."),
    ("c_quadratic_factorization", "c_quadratic_formula", "related", 0.7, "Both are standard solution methods for quadratics."),
    ("c_quadratic_formula", "c_discriminant_nature_roots", "prerequisite", 0.9, "The discriminant arises directly from the quadratic formula."),
    ("c_discriminant_nature_roots", "c_quadratic_applications", "related", 0.65, "Root interpretation helps in contextual quadratic problems."),
    ("c_ap_nth_term", "c_ap_sum", "prerequisite", 0.9, "Nth-term understanding supports deriving AP sums."),
    ("c_ap_sum", "c_ap_applications", "prerequisite", 0.9, "Application problems use the AP sum formula."),
    ("c_distance_formula", "c_section_formula", "related", 0.55, "Both are coordinate-geometry techniques on line segments."),
    ("c_section_formula", "c_area_of_triangle", "related", 0.5, "Coordinate triangle problems often combine section and area reasoning."),
    ("c_similar_triangles", "c_basic_proportionality_theorem", "prerequisite", 1.0, "BPT is grounded in triangle similarity reasoning."),
    ("c_basic_proportionality_theorem", "c_similarity_area_ratio", "prerequisite", 0.8, "Proportional triangle sides support area-ratio reasoning."),
    ("c_pythagoras_theorem", "c_trigonometric_ratios", "prerequisite", 0.7, "Right-triangle side reasoning supports ratio derivation."),
    ("c_tangent_radius_perpendicular", "c_tangents_from_external_point", "prerequisite", 0.85, "The tangent-radius theorem supports tangent-length proofs."),
    ("c_trigonometric_ratios", "c_standard_trig_values", "prerequisite", 1.0, "Standard values are evaluations of trigonometric ratios."),
    ("c_trigonometric_ratios", "c_trig_identities", "prerequisite", 0.95, "Identity work depends on ratio definitions."),
    ("c_trigonometric_ratios", "c_heights_distances", "prerequisite", 1.0, "Heights and distances are applications of trigonometric ratios."),
    ("c_sectors_segments", "c_circle_perimeter_area_applications", "prerequisite", 0.9, "Area and perimeter applications build on sector and segment formulas."),
    ("c_combination_solids_surface_area", "c_combination_solids_volume", "related", 0.55, "Composite-solid problems commonly pair surface-area and volume reasoning."),
    ("c_grouped_data_mean", "c_grouped_data_median_mode", "related", 0.6, "All are central-tendency measures for grouped data."),
]


def syllabus_topic_seed() -> list[SyllabusTopicRecord]:
    return [
        SyllabusTopicRecord(
            topic_id=topic_id,
            topic_name=topic_name,
            class_level=CBSE_CLASS_LEVEL,
            syllabus_session=CBSE_MATH_SESSION,
            unit_id=unit_id,
            unit_name=unit_name,
            unit_marks=unit_marks,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
            periods=periods,
            official_text=official_text,
            source_lines=source_lines,
            source_document=CBSE_MATH_SOURCE_DOCUMENT,
            source_url=CBSE_MATH_SOURCE_URL,
            source_notes=source_notes,
        )
        for topic_id, topic_name, unit_id, unit_name, unit_marks, chapter_id, chapter_name, periods, official_text, source_lines, source_notes in TOPIC_ROWS
    ]


def chapter_and_concept_seed() -> list[ConceptRecord]:
    concept_rows: list[ConceptRecord] = []
    for chapter_id, name, unit_name, unit_marks, periods, order_index, description, learning_objective, tags in CHAPTER_ROWS:
        concept_rows.append(
            ConceptRecord(
                concept_id=chapter_id,
                name=name,
                node_type="chapter",
                chapter_id=chapter_id,
                chapter_name=name,
                parent_concept_id=None,
                class_level=CBSE_CLASS_LEVEL,
                description=description,
                learning_objective=learning_objective,
                tags=tags,
                difficulty_band="core",
                order_index=order_index,
                source_document=CBSE_MATH_SOURCE_DOCUMENT,
                source_url=CBSE_MATH_SOURCE_URL,
                source_kind="official_text",
                syllabus_session=CBSE_MATH_SESSION,
                unit_name=unit_name,
                unit_marks=unit_marks,
                periods=periods,
            )
        )
    chapter_meta = {
        chapter_id: {"unit_name": unit_name, "unit_marks": unit_marks, "periods": periods}
        for chapter_id, _, unit_name, unit_marks, periods, *_ in CHAPTER_ROWS
    }
    for concept_id, name, chapter_id, chapter_name, description, learning_objective, tags, difficulty_band, order_index, source_kind in CONCEPT_ROWS:
        meta = chapter_meta[chapter_id]
        concept_rows.append(
            ConceptRecord(
                concept_id=concept_id,
                name=name,
                node_type="concept",
                chapter_id=chapter_id,
                chapter_name=chapter_name,
                parent_concept_id=chapter_id,
                class_level=CBSE_CLASS_LEVEL,
                description=description,
                learning_objective=learning_objective,
                tags=tags,
                difficulty_band=difficulty_band,
                order_index=order_index,
                source_document=CBSE_MATH_SOURCE_DOCUMENT,
                source_url=CBSE_MATH_SOURCE_URL,
                source_kind=source_kind,
                syllabus_session=CBSE_MATH_SESSION,
                unit_name=meta["unit_name"],
                unit_marks=meta["unit_marks"],
                periods=meta["periods"],
            )
        )
    return concept_rows


def relationship_seed() -> list[RelationshipRecord]:
    return [
        RelationshipRecord(
            source_concept_id=source_concept_id,
            target_concept_id=target_concept_id,
            relation_type=relation_type,
            weight=weight,
            rationale=rationale,
        )
        for source_concept_id, target_concept_id, relation_type, weight, rationale in RELATIONSHIP_ROWS
    ]
