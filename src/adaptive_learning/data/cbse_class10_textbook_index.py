from __future__ import annotations

from adaptive_learning.data.schemas import TextbookSectionRecord


TEXTBOOK_SOURCE_DOCUMENT = "NCERT Mathematics Textbook for Class X (Contents)"
TEXTBOOK_SOURCE_URL = "https://ncert.nic.in/textbook/pdf/jemh1ps.pdf"
TEXTBOOK_CHAPTER_PDF_TEMPLATE = "https://ncert.nic.in/textbook/pdf/jemh1{chapter_number:02d}.pdf"
CLASS_LEVEL = "CBSE_10"

SECTION_ROWS = [
    (1, "ch_real_numbers", "Real Numbers", "1.1", "Introduction", 1, "L0-L35"),
    (1, "ch_real_numbers", "Real Numbers", "1.2", "The Fundamental Theorem of Arithmetic", 2, "L35-L183"),
    (1, "ch_real_numbers", "Real Numbers", "1.3", "Revisiting Irrational Numbers", 5, "L193-L236"),
    (1, "ch_real_numbers", "Real Numbers", "1.4", "Revisiting Rational Numbers and Their Decimal Expansions", 7, "L236-L261"),
    (2, "ch_polynomials", "Polynomials", "2.1", "Introduction to Polynomials", 16, "L262-L270"),
    (2, "ch_polynomials", "Polynomials", "2.2", "Geometrical Meaning of the Zeroes of a Polynomial", 19, "L270-L278"),
    (2, "ch_polynomials", "Polynomials", "2.3", "Relationship between Zeroes and Coefficients of a Polynomial", 24, "L278-L282"),
    (2, "ch_polynomials", "Polynomials", "2.4", "Division Algorithm for Polynomials", 32, "L282-L284"),
    (3, "ch_pair_linear_equations", "Pair of Linear Equations in Two Variables", "3.1", "Introduction", 38, "L284-L285"),
    (3, "ch_pair_linear_equations", "Pair of Linear Equations in Two Variables", "3.2", "Graphical Method of Solution of a Pair of Linear Equations", 40, "L285-L286"),
    (3, "ch_pair_linear_equations", "Pair of Linear Equations in Two Variables", "3.3", "Algebraic Methods of Solving a Pair of Linear Equations", 47, "L286-L286"),
    (4, "ch_quadratic_equations", "Quadratic Equations", "4.1", "Introduction", 38, "L285-L286"),
    (4, "ch_quadratic_equations", "Quadratic Equations", "4.2", "Quadratic Equations", 39, "L286-L287"),
    (4, "ch_quadratic_equations", "Quadratic Equations", "4.3", "Solution of a Quadratic Equation by Factorisation", 42, "L287-L288"),
    (4, "ch_quadratic_equations", "Quadratic Equations", "4.4", "Nature of Roots", 44, "L288-L289"),
    (5, "ch_arithmetic_progressions", "Arithmetic Progressions", "5.1", "Introduction", 49, "L290-L292"),
    (5, "ch_arithmetic_progressions", "Arithmetic Progressions", "5.2", "Arithmetic Progressions", 51, "L292-L293"),
    (5, "ch_arithmetic_progressions", "Arithmetic Progressions", "5.3", "nth Term of an AP", 56, "L293-L294"),
    (5, "ch_arithmetic_progressions", "Arithmetic Progressions", "5.4", "Sum of First n Terms of an AP", 63, "L294-L295"),
    (6, "ch_triangles", "Triangles", "6.1", "Introduction", 73, "L296-L298"),
    (6, "ch_triangles", "Triangles", "6.2", "Similar Figures", 74, "L298-L299"),
    (6, "ch_triangles", "Triangles", "6.3", "Similarity of Triangles", 79, "L299-L300"),
    (6, "ch_triangles", "Triangles", "6.4", "Criteria for Similarity of Triangles", 85, "L300-L301"),
    (7, "ch_coordinate_geometry", "Coordinate Geometry", "7.1", "Introduction", 99, "L302-L304"),
    (7, "ch_coordinate_geometry", "Coordinate Geometry", "7.2", "Distance Formula", 100, "L304-L305"),
    (7, "ch_coordinate_geometry", "Coordinate Geometry", "7.3", "Section Formula", 106, "L305-L306"),
    (8, "ch_trigonometry", "Introduction to Trigonometry", "8.1", "Introduction", 113, "L307-L309"),
    (8, "ch_trigonometry", "Introduction to Trigonometry", "8.2", "Trigonometric Ratios", 114, "L309-L310"),
    (8, "ch_trigonometry", "Introduction to Trigonometry", "8.3", "Trigonometric Ratios of Some Specific Angles", 121, "L310-L311"),
    (8, "ch_trigonometry", "Introduction to Trigonometry", "8.4", "Trigonometric Identities", 128, "L311-L312"),
    (9, "ch_trigonometry", "Introduction to Trigonometry", "9.1", "Heights and Distances", 133, "L313-L315"),
    (10, "ch_circles", "Circles", "10.1", "Introduction", 144, "L316-L318"),
    (10, "ch_circles", "Circles", "10.2", "Tangent to a Circle", 145, "L318-L319"),
    (10, "ch_circles", "Circles", "10.3", "Number of Tangents from a Point on a Circle", 147, "L319-L320"),
    (11, "ch_areas_related_to_circles", "Areas Related to Circles", "11.1", "Areas of Sector and Segment of a Circle", 154, "L321-L323"),
    (12, "ch_surface_areas_volumes", "Surface Areas and Volumes", "12.1", "Introduction", 161, "L325-L327"),
    (12, "ch_surface_areas_volumes", "Surface Areas and Volumes", "12.2", "Surface Area of a Combination of Solids", 162, "L327-L328"),
    (12, "ch_surface_areas_volumes", "Surface Areas and Volumes", "12.3", "Volume of a Combination of Solids", 167, "L328-L329"),
    (13, "ch_statistics", "Statistics", "13.1", "Introduction", 171, "L330-L332"),
    (13, "ch_statistics", "Statistics", "13.2", "Mean of Grouped Data", 171, "L332-L333"),
    (13, "ch_statistics", "Statistics", "13.3", "Mode of Grouped Data", 183, "L333-L334"),
    (13, "ch_statistics", "Statistics", "13.4", "Median of Grouped Data", 188, "L334-L335"),
    (14, "ch_probability", "Probability", "14.1", "Probability — A Theoretical Approach", 202, "L336-L338"),
]


def textbook_section_seed() -> list[TextbookSectionRecord]:
    records: list[TextbookSectionRecord] = []
    for chapter_number, chapter_id, chapter_name, section_number, section_title, start_page, source_lines in SECTION_ROWS:
        records.append(
            TextbookSectionRecord(
                section_id=f"{chapter_id}::{section_number}",
                class_level=CLASS_LEVEL,
                chapter_number=chapter_number,
                chapter_id=chapter_id,
                chapter_name=chapter_name,
                section_number=section_number,
                section_title=section_title,
                start_page=start_page,
                chapter_pdf_url=TEXTBOOK_CHAPTER_PDF_TEMPLATE.format(chapter_number=chapter_number),
                source_document=TEXTBOOK_SOURCE_DOCUMENT,
                source_url=TEXTBOOK_SOURCE_URL,
                source_lines=source_lines,
            )
        )
    return records
