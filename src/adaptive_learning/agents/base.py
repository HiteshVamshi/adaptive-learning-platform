from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from adaptive_learning.agents.schemas import AgentResponse, ToolTrace
from adaptive_learning.agents.tools import AgentToolbox


@runtime_checkable
class Agent(Protocol):
    name: str
    toolbox: AgentToolbox

    def respond(self, user_query: str, **kwargs: Any) -> AgentResponse: ...


@dataclass
class BaseAgent:
    name: str
    toolbox: AgentToolbox

    def respond(self, user_query: str, **kwargs: Any) -> AgentResponse:
        raise NotImplementedError

    def _trace(
        self,
        *,
        traces: list[ToolTrace],
        tool_name: str,
        purpose: str,
        input_summary: dict[str, Any],
        output_summary: dict[str, Any],
    ) -> None:
        traces.append(
            ToolTrace(
                tool_name=tool_name,
                purpose=purpose,
                input_summary=input_summary,
                output_summary=output_summary,
            )
        )
