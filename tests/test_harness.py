"""Script to test the agent harness."""

import json
from unittest.mock import Mock, patch

import pytest

from src.config import config
from src.context_manager import ContextManager
from src.harness import AgentHarness
from src.llm import LLM
from src.tools import ReadFileTool, ToolKit


def test_init() -> None:
    """Test that the harness stores its dependencies and starts with no history."""

    llm = Mock(spec=LLM)
    context_manager = Mock(spec=ContextManager)
    harness = AgentHarness(llm, context_manager)

    assert harness.llm is llm
    assert harness.context_manager is context_manager
    assert harness.history == []
    assert not hasattr(harness, "subagent")


def test_handle_tool_call() -> None:
    """Test that the handler executes a request and records its result."""

    context_manager = Mock(spec=ContextManager)
    context_manager.tool_kit = Mock(spec=ToolKit)
    context_manager.tool_kit.execute.return_value = "file contents"
    harness = AgentHarness(Mock(spec=LLM), context_manager)
    response = '{"tool": "read_file", "arguments": {"path": "README.md"}}'

    assert harness._handle_tool_call(response) is True
    context_manager.tool_kit.execute.assert_called_once_with(
        "read_file", {"path": "README.md"}
    )
    assert harness.history == [
        {"role": "user", "content": "Tool result (read_file):\nfile contents"}
    ]


def test_process_turn_executes_tool() -> None:
    """Test that a tool call is executed and returned to the model."""

    tool = Mock()
    tool.name = "read_file"
    tool.description = "Read a file."
    tool.args = {"path": "File path."}
    tool.execute.return_value = "file contents"
    tool_kit = ToolKit([tool])
    context_manager = ContextManager(
        tool_kit,
        path_rules="tests/data/AGENTS.md",
        path_skills="tests/data/skills",
    )

    tool_call = '{"tool": "read_file", "arguments": {"path": "README.md"}}'
    llm = Mock(spec=LLM)
    llm.chat.side_effect = [tool_call, "Finished."]
    harness = AgentHarness(llm, context_manager)

    harness._process_turn("Read the README.")

    tool.execute.assert_called_once_with(path="README.md")
    assert llm.chat.call_count == 2
    assert harness.history[1] == {"role": "assistant", "content": tool_call}
    assert harness.history[2] == {
        "role": "user",
        "content": "Tool result (read_file):\nfile contents",
    }
    assert harness.history[-1] == {"role": "assistant", "content": "Finished."}


@pytest.mark.parametrize("response", ["Hello!", '{"answer": 42}', "[]", '{"tool":'])
def test_handle_tool_call_without_tool(response: str) -> None:
    """Only a complete tool-request object triggers execution."""

    context_manager = Mock(spec=ContextManager)
    context_manager.tool_kit = Mock(spec=ToolKit)
    harness = AgentHarness(Mock(spec=LLM), context_manager)

    assert harness._handle_tool_call(response) is False
    assert harness.history == []
    context_manager.tool_kit.execute.assert_not_called()


@pytest.mark.parametrize("arguments", ["[]", '{"path": 1}'])
def test_handle_tool_call_invalid_arguments(arguments: str) -> None:
    """Argument errors from the toolkit propagate to the caller."""

    context_manager = ContextManager(ToolKit([ReadFileTool()]))
    harness = AgentHarness(Mock(spec=LLM), context_manager)
    response = '{"tool": "read_file", "arguments": ' + arguments + "}"

    with pytest.raises(TypeError):
        harness._handle_tool_call(response)

    assert harness.history == []


def test_process_turn_propagates_tool_error() -> None:
    """Tool errors stop the turn and propagate to the caller."""

    context_manager = Mock(spec=ContextManager)
    context_manager.get_prefix.return_value = "Instructions"
    context_manager.tool_kit = Mock(spec=ToolKit)
    context_manager.tool_kit.execute.side_effect = FileNotFoundError("missing file")
    llm = Mock(spec=LLM)
    response = '{"tool": "read_file", "arguments": {"path": "missing.txt"}}'
    llm.chat.return_value = response
    harness = AgentHarness(llm, context_manager)

    with pytest.raises(FileNotFoundError, match="missing file"):
        harness._process_turn("Read a file")

    assert harness.history == [
        {"role": "user", "content": "Instructions\n\nRead a file"},
        {"role": "assistant", "content": response},
    ]
    assert llm.chat.call_count == 1


@pytest.mark.parametrize("limit", [0, 2])
def test_process_turn_tool_limit(limit: int) -> None:
    """Bound tool calls, resetting the budget each turn."""

    context_manager = Mock(spec=ContextManager)
    context_manager.get_prefix.return_value = "Instructions"
    context_manager.tool_kit = Mock(spec=ToolKit)
    context_manager.tool_kit.execute.return_value = "result"
    llm = Mock(spec=LLM)
    request = '{"tool": "read_file", "arguments": {"path": "README.md"}}'
    llm.chat.side_effect = [request] * (2 * limit + 1)
    harness = AgentHarness(llm, context_manager)

    with patch.object(config.tools, "max_calls_per_turn", limit):
        for _ in range(2):
            harness._process_turn("Read a file")
            assert harness.history[-1] == {
                "role": "assistant",
                "content": f"Stopped: tool-call limit reached ({limit}).",
            }

    assert context_manager.tool_kit.execute.call_count == 2 * limit
    assert llm.chat.call_count == 2 * limit


def test_handle_tool_call_can_be_interrupted() -> None:
    """Ctrl+C must not be swallowed as a recoverable tool error."""

    context_manager = Mock(spec=ContextManager)
    context_manager.tool_kit = Mock(spec=ToolKit)
    context_manager.tool_kit.execute.side_effect = KeyboardInterrupt
    harness = AgentHarness(Mock(spec=LLM), context_manager)

    with pytest.raises(KeyboardInterrupt):
        harness._handle_tool_call('{"tool": "shell", "arguments": {"command": "ls"}}')

    assert harness.history == []


@pytest.mark.parametrize(
    "queries",
    [["/exit"], ["", "/exit"], ["q", "/exit"], ["Hello", "Again", "/exit"]],
)
def test_run(queries: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """Test exit, blank and ordinary input, and multiple printed responses."""

    turns = len(queries) - 1
    llm = Mock(spec=LLM)
    llm.chat.side_effect = ["Reply"] * turns
    context_manager = Mock(spec=ContextManager)
    context_manager.get_prefix.return_value = "Instructions"
    harness = AgentHarness(llm, context_manager)

    with patch("builtins.input", side_effect=queries):
        harness.run()

    assert llm.chat.call_count == turns
    assert len(harness.history) == 2 * turns
    assert [message["content"] for message in harness.history[::2]] == [
        f"Instructions\n\n{query}" for query in queries[:-1]
    ]
    assert capsys.readouterr().out == "Agent: Reply\n" * turns


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "/help",
            (
                "/help   Show this help message.\n"
                "/compact Summarize the chat history.\n"
                "/reset  Clear the chat history.\n"
                "/exit   Exit the agent.\n"
            ),
        ),
        ("/unknown", "Unknown command. Use /help for available commands.\n"),
    ],
)
def test_run_local_commands(
    command: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Local commands do not reach the model."""

    llm = Mock(spec=LLM)
    harness = AgentHarness(llm, Mock(spec=ContextManager))
    harness.history = [{"role": "user", "content": "Previous message"}]
    previous_history = harness.history.copy()

    with patch("builtins.input", side_effect=[command, "/exit"]):
        harness.run()

    llm.chat.assert_not_called()
    assert harness.history == previous_history
    assert capsys.readouterr().out == expected


def test_compact_history_empty(capsys: pytest.CaptureFixture[str]) -> None:
    """Compacting an empty conversation still stores the model's summary."""

    llm = Mock(spec=LLM)
    llm.chat.return_value = "Nothing to summarize."
    harness = AgentHarness(llm, Mock(spec=ContextManager))

    assert harness._handle_command("/compact") is True

    request = llm.chat.call_args.args[0]
    assert json.loads(request[1]["content"]) == []
    assert harness.history == [
        {"role": "assistant", "content": "Conversation summary:\nNothing to summarize."}
    ]
    assert capsys.readouterr().out == "Chat history compacted.\n"


def test_compact_history_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Model errors during compaction propagate and preserve the history."""

    llm = Mock(spec=LLM)
    llm.chat.side_effect = RuntimeError("Unavailable")
    harness = AgentHarness(llm, Mock(spec=ContextManager))
    original = [{"role": "user", "content": "Fix the tests"}]
    harness.history = original

    with pytest.raises(RuntimeError, match="Unavailable"):
        harness._handle_command("/compact")

    assert harness.history is original
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("response", ["", "   "])
def test_compact_history_empty_summary(
    response: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty model response is stored as an empty summary."""

    llm = Mock(spec=LLM)
    llm.chat.return_value = response
    harness = AgentHarness(llm, Mock(spec=ContextManager))

    assert harness._handle_command("/compact") is True

    assert harness.history == [
        {"role": "assistant", "content": "Conversation summary:\n"}
    ]
    assert capsys.readouterr().out == "Chat history compacted.\n"


def test_run_compact(capsys: pytest.CaptureFixture[str]) -> None:
    """Compact through the command loop, then continue with fresh instructions."""

    llm = Mock(spec=LLM)
    llm.chat.side_effect = ["Tests still need fixing.", "Finished."]
    context_manager = Mock(spec=ContextManager)
    context_manager.get_prefix.return_value = "Instructions"
    harness = AgentHarness(llm, context_manager)
    original = [
        {"role": "user", "content": "Fix the tests"},
        {"role": "assistant", "content": "I found the failure."},
    ]
    harness.history = original.copy()

    with patch("builtins.input", side_effect=["/compact", "Continue", "/exit"]):
        harness.run()

    summary_request = llm.chat.call_args_list[0].args[0]
    assert summary_request[0]["role"] == "system"
    assert json.loads(summary_request[1]["content"]) == original
    expected = [
        {
            "role": "assistant",
            "content": "Conversation summary:\nTests still need fixing.",
        },
        {
            "role": "user",
            "content": "Instructions\n\nContinue",
        },
        {"role": "assistant", "content": "Finished."},
    ]
    assert harness.history == expected
    assert llm.chat.call_count == 2
    assert capsys.readouterr().out == "Chat history compacted.\nAgent: Finished.\n"


@pytest.mark.parametrize("populated", [False, True])
def test_run_reset(populated: bool, capsys: pytest.CaptureFixture[str]) -> None:
    """Reset handles empty history and the next turn starts fresh."""

    llm = Mock(spec=LLM)
    llm.chat.side_effect = ["Reply"]
    context_manager = Mock(spec=ContextManager)
    context_manager.get_prefix.return_value = "Instructions"
    harness = AgentHarness(llm, context_manager)
    if populated:
        harness.history.append({"role": "user", "content": "Previous message"})

    with patch("builtins.input", side_effect=[" /reset ", "Hello", "/exit"]):
        harness.run()

    assert harness.history == [
        {
            "role": "user",
            "content": "Instructions\n\nHello",
        },
        {"role": "assistant", "content": "Reply"},
    ]
    llm.chat.assert_called_once_with(harness.history)
    assert capsys.readouterr().out == "Chat history cleared.\nAgent: Reply\n"


@pytest.mark.parametrize(
    ("command", "keep_running"),
    [("/help", True), ("/reset", True), ("/exit", False), ("/unknown", True)],
)
def test_handle_command(command: str, keep_running: bool) -> None:
    """Only exit stops the loop, and only reset clears history."""

    llm = Mock(spec=LLM)
    harness = AgentHarness(llm, Mock(spec=ContextManager))
    previous_history = [{"role": "user", "content": "Previous message"}]
    harness.history = previous_history.copy()

    assert harness._handle_command(command) is keep_running
    assert harness.history == ([] if command == "/reset" else previous_history)
    llm.chat.assert_not_called()
