from __future__ import annotations

import pandas as pd

from adaptive_learning.agents.base import BaseAgent
from adaptive_learning.agents.schemas import AgentResponse


class TutorAgent(BaseAgent):
    def __init__(self, toolbox) -> None:
        super().__init__(name="TutorAgent", toolbox=toolbox)

    def respond(self, user_query: str, **kwargs) -> AgentResponse:
        traces = []
        live_snapshot = kwargs.get("live_mastery_snapshot")
        if live_snapshot is not None and not isinstance(live_snapshot, pd.DataFrame):
            live_snapshot = None

        search_payload = self.toolbox.hybrid_search(user_query, top_k=3)
        if search_payload.get("_tool_error"):
            return AgentResponse(
                agent_name=self.name,
                user_query=user_query,
                answer=f"Hybrid search failed ({search_payload.get('tool')}): {search_payload.get('message', '')}",
                tool_trace=traces,
                metadata={"tool_error": search_payload},
            )
        self._trace(
            traces=traces,
            tool_name="hybrid_search",
            purpose="Detect likely concepts and supporting practice items for the tutoring query.",
            input_summary={"query": user_query, "top_k": 3},
            output_summary={
                "detected_concepts": search_payload["intent"]["detected_concepts"],
                "result_question_ids": [row["question_id"] for row in search_payload["results"]],
            },
        )

        primary_concept_id = ""
        primary_concept_name = "the requested topic"
        if search_payload["intent"]["detected_concepts"]:
            primary_concept_id = search_payload["intent"]["detected_concepts"][0]
        elif search_payload["results"]:
            primary_concept_id = search_payload["results"][0]["concept_id"]

        mastery_payload = self.toolbox.concept_mastery(primary_concept_id) if primary_concept_id else {}
        if mastery_payload:
            primary_concept_name = str(mastery_payload["concept_name"])
            self._trace(
                traces=traces,
                tool_name="concept_mastery",
                purpose="Tailor the tutoring response to the learner's current mastery state.",
                input_summary={"concept_id": primary_concept_id},
                output_summary={
                    "mastery_band": mastery_payload["mastery_band"],
                    "graph_adjusted_mastery": round(float(mastery_payload["graph_adjusted_mastery"]), 4),
                },
            )

        rag_payload = self.toolbox.rag_answer(user_query, top_k=4)
        if rag_payload.get("_tool_error"):
            return AgentResponse(
                agent_name=self.name,
                user_query=user_query,
                answer=f"RAG tool failed ({rag_payload.get('tool')}): {rag_payload.get('message', '')}",
                tool_trace=traces,
                metadata={"tool_error": rag_payload},
            )
        self._trace(
            traces=traces,
            tool_name="rag_answer",
            purpose="Ground the explanation in retrieved theory and worked-solution context.",
            input_summary={"query": user_query, "top_k": 4},
            output_summary={
                "chunk_ids": [row["chunk_id"] for row in rag_payload["retrieved_chunks"]],
                "answer_preview": rag_payload["answer"][:160],
            },
        )

        neighbors_payload = self.toolbox.concept_neighbors(primary_concept_id) if primary_concept_id else {"neighbors": []}
        if primary_concept_id:
            self._trace(
                traces=traces,
                tool_name="concept_neighbors",
                purpose="Suggest nearby prerequisite or related concepts for follow-up study.",
                input_summary={"concept_id": primary_concept_id},
                output_summary={
                    "neighbor_count": len(neighbors_payload["neighbors"]),
                    "neighbor_concepts": [row["concept_name"] for row in neighbors_payload["neighbors"][:4]],
                },
            )

        mastery_sentence = ""
        if mastery_payload:
            mastery_sentence = (
                f"Your current mastery for {primary_concept_name} is {mastery_payload['mastery_band']} "
                f"(score {float(mastery_payload['graph_adjusted_mastery']):.2f}). "
            )

        follow_up = ""
        if neighbors_payload["neighbors"]:
            names = ", ".join(row["concept_name"] for row in neighbors_payload["neighbors"][:3] if row["concept_name"])
            follow_up = f"Related concepts to review next: {names}."

        learn_sentence = ""
        if primary_concept_id:
            learn_payload = self.toolbox.learn_recommendations(
                live_mastery_snapshot=live_snapshot,
                focus_concept_ids=[primary_concept_id],
                max_concepts=3,
            )
            if learn_payload.get("items"):
                self._trace(
                    traces=traces,
                    tool_name="learn_recommendations",
                    purpose="Surface curated videos and NCERT PDFs for the primary concept.",
                    input_summary={"concept_id": primary_concept_id},
                    output_summary={"item_count": len(learn_payload["items"])},
                )
                learn_sentence = (
                    "\n\n**Learn (curated)**\n"
                    + learn_payload["summary"][:1200]
                    + ("\n…" if len(learn_payload["summary"]) > 1200 else "")
                )

        answer = f"{mastery_sentence}{rag_payload['answer']} {follow_up}{learn_sentence}".strip()
        return AgentResponse(
            agent_name=self.name,
            user_query=user_query,
            answer=answer,
            tool_trace=traces,
            metadata={
                "primary_concept_id": primary_concept_id,
                "primary_concept_name": primary_concept_name,
                "retrieved_chunks": rag_payload["retrieved_chunks"],
            },
        )
