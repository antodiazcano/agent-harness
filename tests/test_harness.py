"""Script to test the agent harness."""

from src.config import config
from src.harness import AgentHarness
from src.llm import LLM


def test_process_turn() -> None:
    """Test for the `_process_turn` function."""

    llm = LLM(config.llm.groq_model)
    agent_harness = AgentHarness(llm)

    query = "hello"
    for i in range(2):
        agent_harness._process_turn(query)
        assert len(agent_harness.history) == 2 * (
            i + 1
        ), "Messages were not appended to the history correctly!"
