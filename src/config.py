"""Configuration of the project."""

from dataclasses import dataclass, field

from groq.types.chat import ChatCompletionMessageParam

type ChatHistory = list[ChatCompletionMessageParam]


@dataclass
class PathsConfig:
    """Class to define the routes to the interesting paths."""

    rules: str = "data/AGENTS.md"
    skills: str = "data/skills"


@dataclass
class LLMConfig:
    """Class to define the configuration of the model."""

    groq_model: str = "openai/gpt-oss-20b"
    temperature: float = 1.0


@dataclass
class ToolsConfig:
    """Configuration for the tools available to the agent."""

    safe_shell_commands: list[str] = field(
        default_factory=lambda: [
            "pwd",
            "ls",
            "find",
            "rg",
            "grep",
            "cat",
            "head",
            "tail",
            "wc",
            "git status",
            "git diff",
        ]
    )


@dataclass
class Config:
    """Main configuration class."""

    paths = PathsConfig()
    llm = LLMConfig()
    tools = ToolsConfig()


config = Config()
