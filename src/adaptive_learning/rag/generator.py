from __future__ import annotations

import re
from dataclasses import dataclass

from adaptive_learning.rag.prompt_templates import build_grounded_prompt
from adaptive_learning.search.text_utils import significant_tokens


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    backend: str
    prompt: str


class BaseAnswerGenerator:
    backend_name: str

    def generate(self, *, query: str, retrieved_chunks: list[dict]) -> GeneratedAnswer:
        raise NotImplementedError


class GroundedSynthesisGenerator(BaseAnswerGenerator):
    backend_name = "grounded_synthesis"
    EXPLANATION_HINTS = {"what", "why", "how", "explain", "define", "meaning", "definition"}

    def generate(self, *, query: str, retrieved_chunks: list[dict]) -> GeneratedAnswer:
        context = "\n\n".join(
            f"[{chunk['chunk_id']}] {chunk['title']}\n{chunk['content']}"
            for chunk in retrieved_chunks
        )
        prompt = build_grounded_prompt(query=query, retrieved_context=context)
        answer = self._compose_answer(query=query, retrieved_chunks=retrieved_chunks)
        return GeneratedAnswer(answer=answer, backend=self.backend_name, prompt=prompt)

    def _compose_answer(self, *, query: str, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return "I could not find enough grounded context to answer this question."

        if self._should_prefer_explanatory_chunks(query):
            theory_answer = self._compose_theory_answer(retrieved_chunks)
            if theory_answer:
                return theory_answer

        for chunk in retrieved_chunks:
            if chunk["chunk_type"] == "solution":
                worked_solution = self._extract_section(
                    chunk["content"], start_label="Worked solution:", end_label="Final answer:"
                )
                final_answer = self._extract_section(
                    chunk["content"], start_label="Final answer:", end_label=None
                )
                parts = []
                if worked_solution:
                    parts.append(worked_solution)
                if final_answer:
                    parts.append(f"Final answer: {final_answer}")
                if parts:
                    return " ".join(parts)

        theory_answer = self._compose_theory_answer(retrieved_chunks)
        if theory_answer:
            return theory_answer

        return self._fallback_sentence_blend(query=query, retrieved_chunks=retrieved_chunks)

    def _compose_theory_answer(self, retrieved_chunks: list[dict]) -> str:
        curriculum_chunk = next(
            (chunk for chunk in retrieved_chunks if chunk["chunk_type"] == "syllabus_textbook_grounded"),
            None,
        )
        if curriculum_chunk:
            summary = self._extract_section(
                curriculum_chunk["content"], start_label="Summary:", end_label="Official syllabus:"
            )
            official_text = self._extract_section(
                curriculum_chunk["content"], start_label="Official syllabus:", end_label="Textbook sections:"
            )
            section_text = self._extract_section(
                curriculum_chunk["content"], start_label="Textbook sections:", end_label=None
            )
            parts = []
            if summary:
                parts.append(summary)
            if official_text:
                parts.append(f"Official syllabus focus: {official_text}")
            if section_text:
                parts.append(f"NCERT section alignment: {section_text}")
            if parts:
                return " ".join(parts)

        theory_chunk = next(
            (chunk for chunk in retrieved_chunks if chunk["chunk_type"] == "theory"),
            None,
        )
        question_chunk = next(
            (chunk for chunk in retrieved_chunks if chunk["chunk_type"] == "question"),
            None,
        )
        if theory_chunk:
            concept_summary = self._extract_section(
                theory_chunk["content"], start_label="Description:", end_label="Learning objective:"
            )
            learning_objective = self._extract_section(
                theory_chunk["content"], start_label="Learning objective:", end_label="Tags:"
            )
            parts = []
            if concept_summary:
                parts.append(concept_summary)
            if learning_objective:
                parts.append(f"Use this idea: {learning_objective}")
            if question_chunk:
                final_answer = self._extract_section(
                    question_chunk["content"], start_label="Expected final answer:", end_label="Difficulty:"
                )
                if final_answer:
                    parts.append(f"Relevant answer form: {final_answer}")
            if parts:
                return " ".join(parts)
        return ""

    def _fallback_sentence_blend(self, *, query: str, retrieved_chunks: list[dict]) -> str:
        query_tokens = set(significant_tokens(query))
        scored_sentences: list[tuple[float, str]] = []

        for chunk in retrieved_chunks:
            chunk_bonus = 0.25 if chunk["chunk_type"] == "solution" else 0.1
            sentences = re.split(r"(?<=[.!?])\s+", chunk["content"])
            for sentence in sentences:
                sentence_tokens = set(significant_tokens(sentence))
                overlap = len(query_tokens.intersection(sentence_tokens))
                if overlap == 0 and chunk["chunk_type"] == "question":
                    continue
                score = float(overlap) + chunk_bonus
                if sentence.strip():
                    scored_sentences.append((score, sentence.strip()))

        top_sentences = []
        seen = set()
        for _, sentence in sorted(scored_sentences, key=lambda item: item[0], reverse=True):
            if sentence in seen:
                continue
            seen.add(sentence)
            top_sentences.append(sentence)
            if len(top_sentences) == 4:
                break

        if not top_sentences:
            top_sentences = [retrieved_chunks[0]["content"]]

        return " ".join(top_sentences)

    def _should_prefer_explanatory_chunks(self, query: str) -> bool:
        query_tokens = set(significant_tokens(query))
        return bool(query_tokens.intersection(self.EXPLANATION_HINTS))

    @staticmethod
    def _extract_section(content: str, *, start_label: str, end_label: str | None) -> str:
        if start_label not in content:
            return ""
        section = content.split(start_label, 1)[1]
        if end_label and end_label in section:
            section = section.split(end_label, 1)[0]
        return section.strip(" .")


class TransformersAnswerGenerator(BaseAnswerGenerator):
    backend_name = "transformers"

    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        from transformers import pipeline

        self.model_name = model_name
        self.pipeline = pipeline(
            task="text2text-generation",
            model=model_name,
            tokenizer=model_name,
        )

    def generate(self, *, query: str, retrieved_chunks: list[dict]) -> GeneratedAnswer:
        context = "\n\n".join(
            f"[{chunk['chunk_id']}] {chunk['title']}\n{chunk['content']}"
            for chunk in retrieved_chunks
        )
        prompt = build_grounded_prompt(query=query, retrieved_context=context)
        output = self.pipeline(prompt, max_new_tokens=180, truncation=True)[0]["generated_text"]
        return GeneratedAnswer(answer=output.strip(), backend=self.backend_name, prompt=prompt)


def build_answer_generator(
    *,
    generator_backend: str = "auto",
    model_name: str = "google/flan-t5-base",
) -> BaseAnswerGenerator:
    if generator_backend == "grounded":
        return GroundedSynthesisGenerator()
    if generator_backend == "transformers":
        return TransformersAnswerGenerator(model_name=model_name)

    try:
        return TransformersAnswerGenerator(model_name=model_name)
    except Exception:
        return GroundedSynthesisGenerator()
