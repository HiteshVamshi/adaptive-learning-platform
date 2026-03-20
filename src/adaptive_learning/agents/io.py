from __future__ import annotations

import json
from pathlib import Path

from adaptive_learning.agents.schemas import AgentResponse


def write_agent_trace(*, output_path: Path, response: AgentResponse) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(response.to_dict(), file_obj, indent=2)
