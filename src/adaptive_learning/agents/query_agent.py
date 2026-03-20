from __future__ import annotations

from adaptive_learning.agents.base import BaseAgent
from adaptive_learning.agents.schemas import AgentResponse


class QueryAgent(BaseAgent):
    def __init__(self, toolbox) -> None:
        super().__init__(name="QueryAgent", toolbox=toolbox)

    def respond(self, user_query: str) -> AgentResponse:
        traces = []

        rewrite_payload = self.toolbox.rewrite_query(user_query)
        self._trace(
            traces=traces,
            tool_name="rewrite_query",
            purpose="Rewrite the raw question into a clearer retrieval-oriented query.",
            input_summary={"query": user_query},
            output_summary={
                "rewritten_query": rewrite_payload["rewritten_query"],
                "detected_concepts": rewrite_payload["intent"]["detected_concepts"],
            },
        )

        primary_concept_id = ""
        if rewrite_payload["intent"]["detected_concepts"]:
            primary_concept_id = rewrite_payload["intent"]["detected_concepts"][0]
        elif rewrite_payload["top_results"]:
            primary_concept_id = rewrite_payload["top_results"][0]["concept_id"]

        neighbor_payload = self.toolbox.concept_neighbors(primary_concept_id) if primary_concept_id else {"neighbors": []}
        if primary_concept_id:
            self._trace(
                traces=traces,
                tool_name="concept_neighbors",
                purpose="Expand the rewritten query with prerequisite and related concepts.",
                input_summary={"concept_id": primary_concept_id},
                output_summary={
                    "neighbor_concepts": [row["concept_name"] for row in neighbor_payload["neighbors"][:4]],
                },
            )

        suggested_terms = [row["concept_name"] for row in neighbor_payload["neighbors"][:3] if row["concept_name"]]
        suggestion_text = ""
        if suggested_terms:
            suggestion_text = f" Add these related concepts if needed: {', '.join(suggested_terms)}."

        answer = (
            f"Rewritten query: {rewrite_payload['rewritten_query']}. "
            f"Top matching questions: "
            + ", ".join(row["question_id"] for row in rewrite_payload["top_results"][:3])
            + "."
            + suggestion_text
        )

        return AgentResponse(
            agent_name=self.name,
            user_query=user_query,
            answer=answer,
            tool_trace=traces,
            metadata={
                "rewritten_query": rewrite_payload["rewritten_query"],
                "intent": rewrite_payload["intent"],
                "top_results": rewrite_payload["top_results"],
            },
        )
