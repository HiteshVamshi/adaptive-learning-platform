from __future__ import annotations

import pandas as pd

from adaptive_learning.agents.base import BaseAgent
from adaptive_learning.agents.schemas import AgentResponse


class LearnAgent(BaseAgent):
    """Suggests learn links (NCERT PDFs, YouTube embeds or search) from learn_resources.csv using mastery-aware selection."""

    def __init__(self, toolbox) -> None:
        super().__init__(name="LearnAgent", toolbox=toolbox)

    def respond(self, user_query: str, **kwargs) -> AgentResponse:
        traces = []
        live_snapshot = kwargs.get("live_mastery_snapshot")
        if live_snapshot is not None and not isinstance(live_snapshot, pd.DataFrame):
            live_snapshot = None

        search_payload = self.toolbox.hybrid_search(user_query, top_k=3)
        focus_ids: list[str] = []
        if search_payload.get("_tool_error"):
            self._trace(
                traces=traces,
                tool_name="hybrid_search",
                purpose="Concept detection skipped after search error.",
                input_summary={"query": user_query},
                output_summary={"error": search_payload.get("message", "")},
            )
        else:
            focus_ids = list(search_payload["intent"].get("detected_concepts") or [])
            if not focus_ids and search_payload.get("results"):
                focus_ids = [search_payload["results"][0].get("concept_id", "")]
            focus_ids = [x for x in focus_ids if x]
            self._trace(
                traces=traces,
                tool_name="hybrid_search",
                purpose="Detect concepts in the learn request for targeted resource matching.",
                input_summary={"query": user_query},
                output_summary={"detected_concepts": focus_ids},
            )

        learn_payload = self.toolbox.learn_recommendations(
            live_mastery_snapshot=live_snapshot,
            focus_concept_ids=focus_ids or None,
            max_concepts=8,
        )
        self._trace(
            traces=traces,
            tool_name="learn_recommendations",
            purpose="Pick videos and slides by mastery band and concept tags.",
            input_summary={"focus_concept_ids": focus_ids, "has_live_snapshot": live_snapshot is not None},
            output_summary={"item_count": len(learn_payload.get("items", []))},
        )

        answer = (
            f"Here are learn links matched to your level and query:\n\n{learn_payload['summary']}\n\n"
            "Open the **Learn** page for embedded video where available; other rows open PDFs or search links."
        )
        return AgentResponse(
            agent_name=self.name,
            user_query=user_query,
            answer=answer.strip(),
            tool_trace=traces,
            metadata={"learn_items": learn_payload.get("items", [])},
        )
