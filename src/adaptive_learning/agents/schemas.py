from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolTrace:
    tool_name: str
    purpose: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    agent_name: str
    user_query: str
    answer: str
    tool_trace: list[ToolTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "user_query": self.user_query,
            "answer": self.answer,
            "tool_trace": [trace.to_dict() for trace in self.tool_trace],
            "metadata": self.metadata,
        }
