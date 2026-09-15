"""Script to test the agent harness."""

from src.config import config
from src.context_manager import ContextManager
from src.harness import AgentHarness
from src.llm import LLM
from src.tools import Tool, ToolKit


def test_process_turn() -> None:
    """Test for the `_process_turn` function."""

    llm = LLM(config.llm.groq_model)
    tools: list[Tool] = []
    tool_kit = ToolKit(tools)
    context_manager = ContextManager(
        tool_kit, path_rules="tests/data/AGENTS.md", path_skills="tests/data/skills"
    )
    agent_harness = AgentHarness(llm, context_manager)

    query = "hello"
    for i in range(2):
        agent_harness._process_turn(query)
        assert len(agent_harness.history) == 2 * (
            i + 1
        ), "Messages were not appended to the history correctly!"
