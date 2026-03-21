"""
Deterministic, curriculum-grounded question generation for every leaf concept.

Produces QuestionRecord + SolutionRecord rows with rich tags (Bloom-style archetype,
cognitive load, template id) so search, RAG, and recommendation can treat them as
first-class items without an external LLM at build time.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from random import Random

import pandas as pd

from adaptive_learning.data.schemas import QuestionRecord, SolutionRecord


@dataclass(frozen=True)
class ConceptBankConfig:
    """Tune volume and reproducibility of the synthetic bank."""

    subject: str  # "math" | "science"
    seed: int = 42
    difficulties: tuple[str, ...] = ("easy", "medium", "hard")
    """Number of distinct templates (prompt families) per difficulty for each concept."""
    templates_per_difficulty: int = 2


def _slug(concept_id: str) -> str:
    return str(concept_id).removeprefix("c_")


def _clean(text: object, max_len: int = 320) -> str:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _theory_snippet(theory_content: pd.DataFrame, concept_id: str) -> str:
    if theory_content is None or theory_content.empty or "concept_id" not in theory_content.columns:
        return ""
    rows = theory_content[theory_content["concept_id"].astype(str) == str(concept_id)]
    if rows.empty:
        return ""
    row = rows.iloc[0]
    for key in ("summary_text", "official_syllabus_text", "title"):
        if key in row.index and pd.notna(row[key]) and str(row[key]).strip():
            return _clean(row[key], 400)
    return ""


def _sibling_concept_name(concepts: pd.DataFrame, concept: dict, rng: Random) -> str:
    cid = str(concept["concept_id"])
    chapter = str(concept["chapter_id"])
    peers = concepts[
        (concepts["node_type"] == "concept")
        & (concepts["chapter_id"].astype(str) == chapter)
        & (concepts["concept_id"].astype(str) != cid)
    ]
    if peers.empty:
        return "another core idea in this chapter"
    row = peers.iloc[rng.randint(0, len(peers) - 1)]
    return str(row["name"])


def _rng_for_concept(global_seed: int, concept_id: str) -> Random:
    h = int(hashlib.md5(str(concept_id).encode("utf-8")).hexdigest()[:8], 16)
    return Random(global_seed ^ h)


def _subject_label(subject: str) -> str:
    return "CBSE Class X Mathematics" if subject == "math" else "CBSE Class X Science (NCERT-aligned)"


def _build_six_templates(
    *,
    concept: dict,
    theory: str,
    sibling_name: str,
    subject: str,
    rng: Random,
) -> list[tuple[str, str, str, str, str, int, list[str]]]:
    """
    Returns list of:
    (template_id, difficulty, question_type, prompt, answer, estimated_time_sec, meta_tags)
    """
    name = str(concept["name"])
    chapter = str(concept["chapter_name"])
    objective = _clean(concept.get("learning_objective"), 280)
    description = _clean(concept.get("description"), 280)
    label = _subject_label(subject)
    theory_bit = f" Hint from course notes: {theory}" if theory else ""

    items: list[tuple[str, str, str, str, str, int, list[str]]] = []

    # Easy — remember / understand
    prompts_e = [
        (
            "recall_def",
            f"In {label}, define **{name}** in your own words (one or two sentences). "
            f"Anchor your answer to how the syllabus describes this topic.",
            f"A concise definition should capture: {description or objective}. "
            f"Official learning direction: {objective}.",
            ["bloom_remember", "archetype_recall", "load_low"],
        ),
        (
            "recognition",
            f"List **three** distinct facts or properties a Class 10 student should remember "
            f"about **{name}** in **{chapter}** (bullet-style in a single paragraph is fine).",
            f"Key points should reflect: {description or objective}. "
            f"Cross-check with the stated learning outcome: {objective}.",
            ["bloom_remember", "archetype_list", "load_low"],
        ),
    ]
    choice_e = rng.sample(range(len(prompts_e)), k=min(2, len(prompts_e)))
    for i, idx in enumerate(sorted(choice_e)):
        _key, pr, ans, meta = prompts_e[idx]
        items.append((f"e{i + 1}", "easy", "short_answer", pr, ans, 95, meta))

    # Medium — understand / analyze
    prompts_m = [
        (
            "explain_link",
            f"Explain how **{name}** supports the bigger picture of **{chapter}** in {label}. "
            f"Use syllabus-level language (no need for exam tricks).",
            f"Strong answers connect the concept to chapter goals. "
            f"Core idea: {description or name}. Expected competence: {objective}.{theory_bit}",
            ["bloom_understand", "archetype_explain", "load_medium"],
        ),
        (
            "theory_hook",
            f"In 4–6 sentences, explain **{name}** so a peer could teach it. "
            f"Mention at least one *why it matters* for {chapter}.",
            f"Explanation should stay faithful to: {objective}. "
            f"Context: {description or chapter}.{theory_bit}",
            ["bloom_understand", "archetype_explain", "load_medium"],
        ),
    ]
    choice_m = rng.sample(range(len(prompts_m)), k=min(2, len(prompts_m)))
    for i, idx in enumerate(sorted(choice_m)):
        _key, pr, ans, meta = prompts_m[idx]
        items.append((f"m{i + 1}", "medium", "conceptual", pr, ans, 165, meta))

    # Hard — apply / connect
    angle = rng.choice(
        [
            f"a practical situation mentioned in the syllabus context for {chapter}",
            f"a typical board-style reasoning chain for {chapter}",
            f"an observation-based prompt you might see in a school lab discussion",
        ]
    )
    prompts_h = [
        (
            "apply",
            f"**Application:** Describe a realistic Class 10 scenario related to **{angle}** "
            f"where **{name}** is the central idea. What reasoning steps would you use?",
            f"Model answer ties the scenario to {objective}. "
            f"Ground reasoning in: {description or theory or objective}.",
            ["bloom_apply", "archetype_scenario", "load_high"],
        ),
        (
            "integrate",
            f"**Connect ideas:** Compare **{name}** with **{sibling_name}** within **{chapter}**. "
            f"What is similar, what is different, and when would you use each?",
            f"Comparison should respect official objectives for both ideas. "
            f"For **{name}**, the syllabus expects: {objective}.",
            ["bloom_analyze", "archetype_compare", "load_high"],
        ),
    ]
    choice_h = rng.sample(range(len(prompts_h)), k=min(2, len(prompts_h)))
    for i, idx in enumerate(sorted(choice_h)):
        _key, pr, ans, meta = prompts_h[idx]
        items.append((f"h{i + 1}", "hard", "application", pr, ans, 255, meta))

    return items


def _question_id(*, subject: str, concept_id: str, template_id: str) -> str:
    sub = "m" if subject == "math" else "s"
    return f"q_bank_{sub}_{_slug(concept_id)}_{template_id}"


def generate_concept_question_bank(
    *,
    subject: str,
    concepts: pd.DataFrame,
    theory_content: pd.DataFrame | None = None,
    config: ConceptBankConfig | None = None,
) -> tuple[list[QuestionRecord], list[SolutionRecord]]:
    """
    Build bank items for every row with node_type == \"concept\".

    Tags encode machine-friendly metadata: subject, chapter slug, difficulty,
    template id, Bloom/archetype/load markers, and ``concept_question_bank`` source.
    """
    normalized_subject = str(subject).strip().lower()
    if config is None:
        cfg = ConceptBankConfig(subject=normalized_subject)
    else:
        cfg = replace(config, subject=normalized_subject)
    concept_rows = concepts[concepts["node_type"] == "concept"].copy()
    concept_rows = concept_rows.sort_values(by=["chapter_id", "order_index", "concept_id"], ignore_index=True)

    chapter_slug = (
        lambda ch: re.sub(r"[^a-z0-9]+", "_", str(ch).lower()).strip("_")[:40] or "chapter"
    )

    q_out: list[QuestionRecord] = []
    s_out: list[SolutionRecord] = []

    for concept in concept_rows.to_dict(orient="records"):
        cid = str(concept["concept_id"])
        rng = _rng_for_concept(cfg.seed, cid)
        theory = _theory_snippet(theory_content if theory_content is not None else pd.DataFrame(), cid)
        sibling = _sibling_concept_name(concept_rows, concept, rng)

        templates = _build_six_templates(
            concept=concept,
            theory=theory,
            sibling_name=sibling,
            subject=normalized_subject,
            rng=rng,
        )

        # Respect templates_per_difficulty: keep first N per difficulty bucket
        by_diff: dict[str, list] = {"easy": [], "medium": [], "hard": []}
        for row in templates:
            by_diff[row[1]].append(row)
        trimmed: list[tuple] = []
        for diff in cfg.difficulties:
            bucket = by_diff.get(diff, [])
            trimmed.extend(bucket[: cfg.templates_per_difficulty])

        for template_id, difficulty, q_type, prompt, answer, time_sec, meta_tags in trimmed:
            qid = _question_id(subject=normalized_subject, concept_id=cid, template_id=template_id)
            ch_slug = chapter_slug(concept["chapter_name"])
            tags = [
                normalized_subject,
                ch_slug,
                difficulty,
                f"tpl_{template_id}",
                "concept_question_bank",
                *meta_tags,
            ]
            worked = (
                f"**Model reasoning:** {answer}\n\n"
                f"**Self-check:** Does your answer state the learning outcome clearly "
                f"and stay within {_subject_label(normalized_subject)} expectations?"
            )
            q_out.append(
                QuestionRecord(
                    question_id=qid,
                    chapter_id=str(concept["chapter_id"]),
                    chapter_name=str(concept["chapter_name"]),
                    concept_id=cid,
                    concept_name=str(concept["name"]),
                    difficulty=difficulty,
                    question_type=q_type,
                    prompt=prompt,
                    answer=_clean(answer, 2000),
                    tags=tags,
                    estimated_time_sec=time_sec,
                    source="concept_question_bank",
                )
            )
            s_out.append(
                SolutionRecord(
                    question_id=qid,
                    final_answer=_clean(answer, 1200),
                    worked_solution=_clean(worked, 4000),
                    explanation_style="concept_question_bank",
                )
            )

    return q_out, s_out


def concept_bank_questions_to_frames(
    *,
    subject: str,
    concepts: pd.DataFrame,
    theory_content: pd.DataFrame | None = None,
    config: ConceptBankConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: same as generate_concept_question_bank but returns DataFrames."""
    qr, sr = generate_concept_question_bank(
        subject=subject,
        concepts=concepts,
        theory_content=theory_content,
        config=config,
    )
    qdf = pd.DataFrame([r.to_dict() for r in qr]).sort_values(
        by=["chapter_id", "concept_id", "difficulty", "question_id"], ignore_index=True
    )
    sdf = pd.DataFrame([r.to_dict() for r in sr]).sort_values(by=["question_id"], ignore_index=True)
    return qdf, sdf
