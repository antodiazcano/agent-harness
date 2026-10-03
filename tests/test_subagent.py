"""Tests for delegation with a mocked model and the real harness."""

from unittest.mock import Mock, patch

import pytest

from src.config import config
from src.context_manager import ContextManager
from src.harness import AgentHarness
from src.llm import LLM
from src.tools import ReadFileTool, ShellTool, ToolKit


def test_delegate_uses_fresh_history() -> None:
    """Every delegated task gets its own history and project instructions."""

    llm = Mock(spec=LLM)
    llm.chat.side_effect = ["First result", "Second result"]
    context = ContextManager(ToolKit([]), "tests/data/AGENTS.md", "tests/data/skills")
    parent = AgentHarness(llm, context)

    assert parent._handle_delegation('{"delegate": "First task"}') is True
    assert parent._handle_delegation('{"delegate": "Second task"}') is True
    assert parent.history == [
        {"role": "user", "content": "Subagent result:\nFirst result"},
        {"role": "user", "content": "Subagent result:\nSecond result"},
    ]

    first = llm.chat.call_args_list[0].args[0]
    second = llm.chat.call_args_list[1].args[0]
    assert first is not second
    assert len(first) == len(second) == 2
    assert first[0]["content"].endswith("\n\nFirst task")
    assert second[0]["content"].endswith("\n\nSecond task")
    assert "This is a dummy file." in first[0]["content"]
    assert "dummy 1" in first[0]["content"]
    assert "Name: delegate" not in first[0]["content"]
    assert first[0]["content"].count(config.prompts.delegation) == 1


def test_parent_receives_child_result() -> None:
    """A child uses a real tool; only its final result reaches the parent."""

    llm = Mock(spec=LLM)
    llm.chat.side_effect = [
        '{"delegate": "Read the rules"}',
        '{"tool": "read_file", "arguments": {"path": "tests/data/AGENTS.md"}}',
        "The rules contain a dummy example.",
        "Parent finished.",
    ]
    context = ContextManager(
        ToolKit([ReadFileTool()]), "tests/data/AGENTS.md", "tests/data/skills"
    )
    parent = AgentHarness(llm, context)

    parent._process_turn("Review this project")

    assert llm.chat.call_count == 4
    assert parent.history[2] == {
        "role": "user",
        "content": "Subagent result:\nThe rules contain a dummy example.",
    }
    assert parent.history[-1]["content"] == "Parent finished."
    child_history = llm.chat.call_args_list[1].args[0]
    assert child_history is not parent.history
    assert "Review this project" not in child_history[0]["content"]
    assert "Name: delegate" not in child_history[0]["content"]
    assert parent.history[0]["content"].count(config.prompts.delegation) == 1
    assert child_history[0]["content"].count(config.prompts.delegation) == 1
    assert [tool.name for tool in context.tool_kit.tools] == ["read_file"]
    assert child_history[2]["content"].startswith("Tool result (read_file):\n")


def test_delegate_can_delegate_again() -> None:
    """Nested delegation keeps histories separate and returns results up the chain."""

    llm = Mock(spec=LLM)
    llm.chat.side_effect = [
        '{"delegate": "Child task"}',
        '{"delegate": "Nested task"}',
        "Nested result",
        "Child result",
        "Parent finished",
    ]
    context = ContextManager(ToolKit([]), "tests/data/AGENTS.md", "tests/data/skills")
    parent = AgentHarness(llm, context)

    parent._process_turn("Parent task")

    assert llm.chat.call_count == 5
    child_history = llm.chat.call_args_list[1].args[0]
    nested_history = llm.chat.call_args_list[2].args[0]
    assert child_history is not parent.history
    assert nested_history is not child_history
    assert nested_history is not parent.history
    assert child_history[0]["content"] == context.get_prefix() + "\n\nChild task"
    assert nested_history[0]["content"] == context.get_prefix() + "\n\nNested task"
    assert child_history[2]["content"] == "Subagent result:\nNested result"
    assert parent.history[2]["content"] == "Subagent result:\nChild result"
    assert parent.history[-1]["content"] == "Parent finished"


def test_delegate_keeps_shell_permissions_and_limit() -> None:
    """A denied shell command stays denied and the child respects its call limit."""

    llm = Mock(spec=LLM)
    llm.chat.side_effect = ['{"tool": "shell", "arguments": {"command": "echo hello"}}']
    context = ContextManager(
        ToolKit([ShellTool()]), "tests/data/AGENTS.md", "tests/data/skills"
    )
    parent = AgentHarness(llm, context)

    with (
        patch.object(config.tools, "max_calls_per_turn", 1),
        patch("builtins.input", return_value="no") as ask_permission,
        patch("src.tools.subprocess.run") as run,
    ):
        assert parent._handle_delegation('{"delegate": "Run echo hello"}') is True

    ask_permission.assert_called_once()
    run.assert_not_called()
    assert parent.history[-1]["content"] == (
        "Subagent result:\nStopped: tool-call limit reached (1)."
    )
    assert llm.chat.call_count == 1
    child_history = llm.chat.call_args.args[0]
    assert "permission denied" in child_history[2]["content"]


def test_delegation_counts_toward_parent_limit() -> None:
    """Delegation consumes the same per-turn budget as ordinary tools."""

    llm = Mock(spec=LLM)
    llm.chat.side_effect = ['{"delegate": "Task"}', "Child finished."]
    context = ContextManager(ToolKit([]), "tests/data/AGENTS.md", "tests/data/skills")
    parent = AgentHarness(llm, context)

    with patch.object(config.tools, "max_calls_per_turn", 1):
        parent._process_turn("Delegate the task")

    assert llm.chat.call_count == 2
    assert parent.history[-2]["content"] == "Subagent result:\nChild finished."
    assert parent.history[-1]["content"] == "Stopped: tool-call limit reached (1)."


@pytest.mark.parametrize(
    "response",
    [
        "Hello!",
        "[]",
        '{"delegate":',
        '{"tool": "read_file", "arguments": {"path": "README.md"}}',
        '{"delegate": "Task", "extra": true}',
    ],
)
def test_handle_delegation_without_request(response: str) -> None:
    """Ordinary responses and tool calls do not start a child."""

    llm = Mock(spec=LLM)
    parent = AgentHarness(llm, Mock(spec=ContextManager))

    assert parent._handle_delegation(response) is False
    llm.chat.assert_not_called()
    assert parent.history == []
