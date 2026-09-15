"""Defines the agent harness and uses it."""

from src.config import config
from src.context_manager import ContextManager
from src.harness import AgentHarness
from src.llm import LLM
from src.tools import Tool, ToolKit


def main() -> None:
    """Defines all the agent parameters and uses it."""

    llm = LLM(config.llm.groq_model, temperature=config.llm.temperature)
    tools: list[Tool] = []
    tool_kit = ToolKit(tools)
    context_manager = ContextManager(
        tool_kit, path_rules=config.paths.rules, path_skills=config.paths.skills
    )

    agent_harness = AgentHarness(llm, context_manager)
    agent_harness.run()


if __name__ == "__main__":
    main()
