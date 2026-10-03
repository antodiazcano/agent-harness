"""Script to test the context manager."""

from src.config import config
from src.context_manager import ContextManager
from src.tools import ReadFileTool, Tool, ToolKit

TOOLS: list[Tool] = []
TK = ToolKit(TOOLS)
PATH_RULES = "tests/data/AGENTS.md"
PATH_SKILLS = "tests/data/skills"

CM = ContextManager(TK, path_rules=PATH_RULES, path_skills=PATH_SKILLS)


def test_get_system_prompt() -> None:
    """Test for the `_get_system_prompt` function."""

    prompt = CM._get_system_prompt()
    expected = f"{config.prompts.agent}\n\n{config.prompts.delegation}"

    assert prompt == expected, "Incorrect prompt!"


def test_get_rules() -> None:
    """Test for the `_get_rules` function."""

    rules = CM._get_rules()
    expected = "This is a dummy file.\n\n# bla bla bla\n$1 + 1 = 2$\n"

    assert rules == expected, "Incorrect rules!"


def test_get_skills_summary() -> None:
    """Test for the `_get_skills_summary` function."""

    skills_summary = CM._get_skills_summary()
    expected = (
        "These are the skills you can use:\n"
        r"- dummy 1: Bla bla\n bla bla bla" + "\n"
        "- dummy2: Bla bla"
    )

    assert skills_summary == expected, "Incorrect skills summary!"


def test_get_tools_summary_empty_toolkit() -> None:
    """Test `_get_tools_summary` with an empty toolkit."""

    tools_summary = CM._get_tools_summary()
    expected = "These are the tools you can use: "

    assert tools_summary == expected, "Incorrect tools summary!"


def test_get_tools_summary_non_empty_toolkit() -> None:
    """Test `_get_tools_summary` with a non-empty toolkit."""

    context_manager = ContextManager(ToolKit([ReadFileTool()]))
    tools_summary = context_manager._get_tools_summary()
    expected = (
        "These are the tools you can use: "
        "Name: read_file\n"
        "Description: Read a UTF-8 file.\n"
        "Args: {'path': 'File path.'}\n\n"
    )

    assert tools_summary == expected, "Incorrect tools summary!"


def test_get_prefix() -> None:
    """Test for the `get_prefix` function."""

    prefix = CM.get_prefix()
    expected = (
        f"{CM._get_system_prompt()}\n\n"
        f"{CM._get_rules()}\n\n"
        f"{CM._get_skills_summary()}\n\n"
        f"{CM._get_tools_summary()}"
    )

    assert prefix == expected, "Incorrect prefix!"
