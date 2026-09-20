"""Script to build the tools the agent can use."""

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from src.config import config


class Tool(ABC):
    """Base class for a tool."""

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
    def execute(self, *args, **kwargs) -> str:
        """Executes the tool.

        Returns:
            Output result.
        """


class ReadFileTool(Tool):
    """Read a UTF-8 file, resolving relative paths from a working directory."""

    def __init__(self, working_directory: str | Path = ".") -> None:
        """Constructor of the class.

        Args:
            working_directory: Current working directory.
        """

        super().__init__("read_file", "Read a UTF-8 file.", {"path": "File path."})

        self.working_directory = Path(working_directory).resolve()

    def execute(self, *, path: str) -> str:
        """Executes the tool.

        Args:
            path: Path to the file.

        Returns:
            Output result.
        """

        return (self.working_directory / path).read_text(encoding="utf-8")


class WriteFileTool(Tool):
    """Create or overwrite a UTF-8 file; its parent directory must exist."""

    def __init__(self, working_directory: str | Path = ".") -> None:
        """Constructor of the class.

        Args:
            working_directory: Current working directory.
        """

        super().__init__(
            "write_file",
            "Create or overwrite a UTF-8 file. Parent directory must exist.",
            {"path": "File path.", "content": "Full file contents."},
        )

        self.working_directory = Path(working_directory).resolve()

    def execute(self, *, path: str, content: str) -> str:
        """Executes the tool.

        Args:
            path: Path to the file.
            content: Content to write.

        Returns:
            Output result.
        """

        (self.working_directory / path).write_text(content, encoding="utf-8")

        return f"Wrote {path}"


class ShellTool(Tool):
    """Run shell commands with the current user's permissions, without a sandbox."""

    def __init__(
        self, working_directory: str | Path = ".", timeout: float = 30
    ) -> None:
        """Constructor of the class.

        Args:
            working_directory: Current working directory.
            timeout: Maximum execution time in seconds.
        """

        super().__init__(
            "shell", "Run a shell command.", {"command": "Shell command to execute."}
        )

        self.working_directory = Path(working_directory).resolve()
        self.timeout = timeout

    @staticmethod
    def _is_safe(command: str) -> bool:
        """Function to decide if a command is safe or not.

        Args:
            command: Command to execute.

        Returns:
            `True` if the command is safe, `False` otherwise.
        """

        return command in config.tools.safe_shell_commands

    def execute(self, *, command: str) -> str:
        """Executes the tool.

        Args:
            command: Command to execute.

        Returns:
            Output result.
        """

        if not self._is_safe(command):
            permission = (
                input(f"Allow shell command `{command}`? [y/n] ").strip().lower()
            )
            if permission not in {"y", "yes"}:
                return "Command not executed: permission denied."

        result = subprocess.run(
            command,
            shell=True,
            cwd=self.working_directory,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )

        return (
            f"Exit code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


class ToolKit:
    """Class to define the set of tools used."""

    def __init__(self, tools: list[Tool]) -> None:
        """Constructor of the class.

        Args:
            tools: List with the available tools to use.
        """

        self.tools = tools

    def execute(self, name: str, arguments: dict[str, str]) -> str:
        """Dispatch to a named tool; execution errors propagate to the caller.

        Args:
            name: Name of the tool.
            arguments: Arguments for the tool.

        Returns:
            Result of the tool.

        Raises:
            ValueError: If the tool name was incorrect.
        """

        for tool in self.tools:
            if tool.name == name:
                return tool.execute(**arguments)

        raise ValueError(f"Unknown tool: {name}")

    def get_summary(self) -> str:
        """Obtains a summary of the tools that can be used.

        Returns:
            Summary for the tools of the kit.
        """

        summary = ""

        for tool in self.tools:
            summary_tool = (
                f"Name: {tool.name}\nDescription: {tool.description}\nArgs: {tool.args}"
            )
            summary += f"{summary_tool}\n\n"

        return summary
