"""Script to test the LLM."""

import json
from unittest.mock import Mock, patch

import pytest

from src.config import ChatHistory, config
from src.llm import LLM
from src.tools import ReadFileTool, ShellTool, WriteFileTool


def test_chat_normal() -> None:
    """Test the LLM chat function."""

    with patch("src.llm.Groq") as groq:
        message = Mock(content="Mock response", tool_calls=None)
        groq.return_value.chat.completions.create.return_value = Mock(
            choices=[Mock(message=message)]
        )
        llm = LLM(config.llm.groq_model)

        history: ChatHistory = [
            {"role": "system", "content": "This is a dummy prompt."}
        ]
        response = llm.chat(history)

        assert response == "Mock response", "Incorrect response!"
        groq.return_value.chat.completions.create.assert_called_once_with(
            messages=history,
            model=config.llm.groq_model,
            temperature=config.llm.temperature,
            tools=None,
            parallel_tool_calls=False,
        )


def test_chat_raises_error_when_content_is_none() -> None:
    """Test of the LLM chat function when the response has no content."""

    with patch("src.llm.Groq") as groq:
        groq.return_value.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=None, tool_calls=None))]
        )
        llm = LLM(config.llm.groq_model)
        history: ChatHistory = [
            {"role": "system", "content": "This is a dummy prompt."}
        ]

        with pytest.raises(RuntimeError):
            llm.chat(history)


def test_chat_returns_tool_request_as_text() -> None:
    """Test that tool-request JSON is passed through as ordinary text."""

    with patch("src.llm.Groq") as groq:
        response = '{"tool": "read_file", "arguments": {"path": "README.md"}}'
        groq.return_value.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=response, tool_calls=None))]
        )
        llm = LLM(config.llm.groq_model)
        history: ChatHistory = [{"role": "user", "content": "Read a file."}]

        assert llm.chat(history) == response


def test_chat_declares_and_normalizes_native_tool_call() -> None:
    """Test that Groq receives the tools and returns harness JSON."""

    with patch("src.llm.Groq") as groq:
        function = Mock(
            name="shell_function",
            arguments='{"command": ".venv/bin/python fibonacci.py 14"}',
        )
        function.name = "shell"
        tool_call = Mock(function=function)
        message = Mock(content=None, tool_calls=[tool_call])
        groq.return_value.chat.completions.create.return_value = Mock(
            choices=[Mock(message=message)]
        )
        tools = [ReadFileTool(), WriteFileTool(), ShellTool()]
        llm = LLM(config.llm.groq_model, tools=tools)
        history: ChatHistory = [{"role": "user", "content": "Run Fibonacci."}]

        assert json.loads(llm.chat(history)) == {
            "tool": "shell",
            "arguments": {"command": ".venv/bin/python fibonacci.py 14"},
        }
        request_tools = groq.return_value.chat.completions.create.call_args.kwargs[
            "tools"
        ]
        assert [tool["function"]["name"] for tool in request_tools] == [
            "read_file",
            "write_file",
            "shell",
            "delegate",
        ]


def test_chat_declares_and_normalizes_native_delegation() -> None:
    """Test that Groq delegation remains separate from harness tools."""

    with patch("src.llm.Groq") as groq:
        task = "Review fibonacci.py for correctness and edge cases."
        function = Mock(arguments=json.dumps({"delegate": task}))
        function.name = "delegate"
        message = Mock(content=None, tool_calls=[Mock(function=function)])
        groq.return_value.chat.completions.create.return_value = Mock(
            choices=[Mock(message=message)]
        )
        llm = LLM(config.llm.groq_model, tools=[ReadFileTool()])

        assert json.loads(llm.chat([{"role": "user", "content": task}])) == {
            "delegate": task
        }
        request_tools = groq.return_value.chat.completions.create.call_args.kwargs[
            "tools"
        ]
        assert [tool["function"]["name"] for tool in request_tools] == [
            "read_file",
            "delegate",
        ]
