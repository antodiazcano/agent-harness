"""Script to build the tools the agent can use."""

from abc import ABC, abstractmethod


class Tool(ABC):
    """Interface for a tool."""

    def __init__(self, name: str, description: str, args: dict[str, str]) -> None:
        """Constructor of the class.

        Args:
            name: Name of the tool.
            description: Description of the tool.
            args: Dictionary mapping each argument to its purpose.
        """

        self.name = name
        self.description = description
        self.args = args

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """Executes the tool."""


class ToolKit:
    """Class to define the set of tools used."""

    def __init__(self, tools: list[Tool]) -> None:
        """Constructor of the class.

        Args:
            tools: List with the available tools to use.
        """

        self.tools = tools

    # def execute(self) ->

    def get_summary(self) -> str:
        """Obtains a summary of the tools that can be used."""

        summary = ""

        for tool in self.tools:
            summary_tool = (
                f"Name: {tool.name}\nDescription: {tool.description}\nArgs: {tool.args}"
            )
            summary += f"{summary_tool}\n\n"

        return summary
