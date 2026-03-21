from __future__ import annotations

from adaptive_learning.agents.base import BaseAgent
from adaptive_learning.agents.schemas import AgentResponse


class PracticeAgent(BaseAgent):
    def __init__(self, toolbox) -> None:
        super().__init__(name="PracticeAgent", toolbox=toolbox)

    def respond(self, user_query: str, **kwargs) -> AgentResponse:
        traces = []

        rewrite_payload = self.toolbox.rewrite_query(user_query)
        if rewrite_payload.get("_tool_error"):
            return AgentResponse(
                agent_name=self.name,
                user_query=user_query,
                answer=f"Rewrite tool failed ({rewrite_payload.get('tool')}): {rewrite_payload.get('message', '')}",
                tool_trace=traces,
                metadata={"tool_error": rewrite_payload},
            )
        self._trace(
            traces=traces,
            tool_name="rewrite_query",
            purpose="Normalize the practice request into explicit concepts and difficulty signals.",
            input_summary={"query": user_query},
            output_summary={
                "rewritten_query": rewrite_payload["rewritten_query"],
                "detected_concepts": rewrite_payload["intent"]["detected_concepts"],
            },
        )

        weak_concepts = self.toolbox.weak_concepts(top_k=5)
        self._trace(
            traces=traces,
            tool_name="weak_concepts",
            purpose="Prioritize concepts that currently need support.",
            input_summary={"top_k": 5},
            output_summary={
                "concept_ids": [row["concept_id"] for row in weak_concepts],
                "bands": [row["mastery_band"] for row in weak_concepts],
            },
        )

        target_concept_id = ""
        if rewrite_payload["intent"]["detected_concepts"]:
            target_concept_id = rewrite_payload["intent"]["detected_concepts"][0]
        elif weak_concepts:
            target_concept_id = weak_concepts[0]["concept_id"]

        concept_mastery = self.toolbox.concept_mastery(target_concept_id) if target_concept_id else {}
        target_difficulty = "medium"
        if concept_mastery:
            target_difficulty = {
                "needs_support": "easy",
                "developing": "medium",
                "mastered": "hard",
            }[concept_mastery["mastery_band"]]
            self._trace(
                traces=traces,
                tool_name="concept_mastery",
                purpose="Choose a difficulty progression that matches current skill.",
                input_summary={"concept_id": target_concept_id},
                output_summary={
                    "mastery_band": concept_mastery["mastery_band"],
                    "target_difficulty": target_difficulty,
                },
            )

        recommendation_rows = self.toolbox.recommendation_list(top_k=3, concept_id=target_concept_id or None)
        if not recommendation_rows:
            recommendation_rows = self.toolbox.recommendation_list(top_k=3)
        self._trace(
            traces=traces,
            tool_name="recommendation_list",
            purpose="Fetch the highest-value existing questions for the learner right now.",
            input_summary={"top_k": 3, "concept_id": target_concept_id or None},
            output_summary={"question_ids": [row["question_id"] for row in recommendation_rows]},
        )

        generated_payload = {}
        if target_concept_id:
            generated_payload = self.toolbox.generate_question(target_concept_id, target_difficulty)
            self._trace(
                traces=traces,
                tool_name="generate_question",
                purpose="Create an additional targeted practice question for the chosen concept.",
                input_summary={"concept_id": target_concept_id, "difficulty": target_difficulty},
                output_summary={"generated_prompt": generated_payload["output"]["prompt"]},
            )

        recommendation_text = " ".join(
            f"{index+1}. {row['question_id']} on {row['concept_name']} ({row['difficulty']})"
            for index, row in enumerate(recommendation_rows)
        )
        generated_text = ""
        if generated_payload:
            generated = generated_payload["output"]
            generated_text = (
                f" Generated practice: {generated['prompt']} Answer: {generated['final_answer']} "
                f"Explanation: {generated['explanation']}"
            )

        answer = (
            f"Recommended practice sequence: {recommendation_text}. "
            f"Focus concept: {concept_mastery.get('concept_name', target_concept_id)} at {target_difficulty} difficulty."
            f"{generated_text}"
        ).strip()

        return AgentResponse(
            agent_name=self.name,
            user_query=user_query,
            answer=answer,
            tool_trace=traces,
            metadata={
                "target_concept_id": target_concept_id,
                "recommendations": recommendation_rows,
                "generated_question": generated_payload.get("output"),
            },
        )
