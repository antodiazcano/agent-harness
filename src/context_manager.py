"""Script to define the manager of the context."""

import re
from pathlib import Path

from src.config import config
from src.tools import ToolKit


class ContextManager:
    """Class to manage the context of the agent harness."""

    def __init__(
        self,
        tool_kit: ToolKit,
        path_rules: str = config.paths.rules,
        path_skills: str = config.paths.skills,
    ) -> None:
        """Constructor of the class.

        Args:
            path_rules: Path to `AGENTS.md`
            path_skill: Path to the folder where the skills are saved.
            tools: Tools that can be used.
        """

        self.tool_kit = tool_kit
        self.path_rules = path_rules
        self.path_skills = path_skills

    @staticmethod
    def _get_system_prompt() -> str:
        """Returns the system prompt.

        Returns:
            System prompt.
        """

        return "You are Tony, a small local coding agent."

    def _get_rules(self) -> str:
        """Returns the rules (`AGENTS.md`) for the agent.

        Returns:
            Content of the `AGENTS.md` file.
        """

        with open(self.path_rules, "r", encoding="utf-8") as f:
            return f.read()

    def _get_skills_summary(self) -> str:
        """Obtains a summary of the skills the agent can use.

        Returns:
            Summary of the skills that can be used.
        """

        skills = ["These are the skills you can use:"]
        for skill_path in sorted(Path(self.path_skills).glob("*.md")):
            content = skill_path.read_text(encoding="utf-8")
            match = re.search(
                r"^name:\s*(.+)$[\s\S]*?^description:\s*[>|]-?\s*$\n"
                r"((?:(?:[ \t]+[^\n]*)?\n)*)",
                content,
                re.MULTILINE,
            )
            if match:
                name, description = match.groups()
                skills.append(f"- {name}: {' '.join(description.split())}")

        return "\n".join(skills)

    def _get_tools_summary(self) -> str:
        """Obtains a summary of the skills the agent can use.

        Returns:
            Summary of the skills that can be used.
        """

        return f"These are the tools you can use: {self.tool_kit.get_summary()}"

    def get_prefix(self) -> str:
        """Obtains the prefix. for the prompt.

        Returns:
            Prefix for the prompt.
        """

        return (
            f"{self._get_system_prompt()}\n\n"
            f"{self._get_rules()}\n\n"
            f"{self._get_skills_summary()}\n\n"
            f"{self._get_tools_summary()}"
        )
