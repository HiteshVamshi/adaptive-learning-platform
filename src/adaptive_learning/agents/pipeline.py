from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adaptive_learning.agents.practice_agent import PracticeAgent
from adaptive_learning.agents.query_agent import QueryAgent
from adaptive_learning.agents.tools import AgentToolbox
from adaptive_learning.agents.tutor_agent import TutorAgent


@dataclass(frozen=True)
class AgentSuite:
    tutor_agent: TutorAgent
    practice_agent: PracticeAgent
    query_agent: QueryAgent

    def get(self, agent_name: str):
        mapping = {
            "tutor": self.tutor_agent,
            "practice": self.practice_agent,
            "query": self.query_agent,
        }
        return mapping[agent_name]


def build_agent_suite(
    *,
    data_dir: Path,
    search_index_dir: Path,
    rag_index_dir: Path,
    mastery_dir: Path,
    recommendation_dir: Path,
) -> AgentSuite:
    toolbox = AgentToolbox(
        data_dir=data_dir,
        search_index_dir=search_index_dir,
        rag_index_dir=rag_index_dir,
        mastery_dir=mastery_dir,
        recommendation_dir=recommendation_dir,
    )
    return AgentSuite(
        tutor_agent=TutorAgent(toolbox),
        practice_agent=PracticeAgent(toolbox),
        query_agent=QueryAgent(toolbox),
    )
