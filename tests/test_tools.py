"""Tests for file tools, shell execution, and tool dispatch."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import config
from src.tools import ReadFileTool, ShellTool, ToolKit, WriteFileTool

# Read and write tools


def test_write_and_read_file_tools() -> None:
    """Test to check the tool for writing and reading files."""

    writing = WriteFileTool()
    reading = ReadFileTool()

    path = "tests/dummy.txt"

    try:
        assert not os.path.exists(path)
        content = "Hello world!\n This is a test."
        out = writing.execute(path=path, content=content)
        assert out == f"Wrote {path}"
        assert os.path.exists(path)
        text_read = reading.execute(path=path)
        assert text_read == content
    finally:
        if os.path.exists(path):
            os.remove(path)


# Shell tool


def test_is_safe() -> None:
    """Test for the `_is_safe` function."""

    tool = ShellTool()

    # Safe command
    command = "ls"
    assert tool._is_safe(command)

    # Unsafe command
    command = "rm"
    assert not tool._is_safe(command)


def test_shell(tmp_path: Path) -> None:
    """Test for the `execute` function."""

    toolkit = ToolKit([ShellTool(tmp_path)])
    with patch("builtins.input", return_value="yes") as ask_permission:
        result = toolkit.execute("shell", {"command": "pwd; printf hello | tr a-z A-Z"})
    assert "Exit code: 0\n" in result
    assert str(tmp_path) in result
    assert "HELLO" in result
    ask_permission.assert_called_once()

    with patch("builtins.input", return_value="y"):
        result = toolkit.execute("shell", {"command": "printf failed >&2; exit 3"})
    assert "Exit code: 3\n" in result
    assert "stderr:\nfailed" in result


def test_safe_shell_command_does_not_ask_permission(tmp_path: Path) -> None:
    """Test to check a safe command does not ask for permission."""

    with patch("builtins.input") as ask_permission:
        result = ShellTool(tmp_path).execute(command="pwd")

    assert "Exit code: 0\n" in result
    ask_permission.assert_not_called()
    assert "pwd" in config.tools.safe_shell_commands


def test_unsafe_shell_command_can_be_rejected(tmp_path: Path) -> None:
    """Test to check an unsafe command asks for permission."""

    with (
        patch("builtins.input", return_value="no"),
        patch("src.tools.subprocess.run") as run,
    ):
        result = ShellTool(tmp_path).execute(command="touch rejected.txt")

    assert result == "Command not executed: permission denied."
    run.assert_not_called()
    assert not (tmp_path / "rejected.txt").exists()


def test_shell_timeout(tmp_path: Path) -> None:
    """Test to check the timeout for a command works."""

    with (
        patch("builtins.input", return_value="yes"),
        patch(
            "src.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired("sleep 10", 0.1),
        ) as run,
    ):
        with pytest.raises(subprocess.TimeoutExpired):
            ShellTool(tmp_path, timeout=0.1).execute(command="sleep 10")
        assert run.call_args.kwargs["timeout"] == 0.1


# Toolkit tests


def test_execute() -> None:
    """Test for `execute` function."""

    tool_kit = ToolKit([ReadFileTool(), WriteFileTool()])
    path = "tests/dummy.txt"

    try:
        assert not os.path.exists(path)
        content = "Hello world!\n This is a test."
        out = tool_kit.execute(
            "write_file", arguments={"path": path, "content": content}
        )
        assert out == f"Wrote {path}"
        assert os.path.exists(path)
        text_read = tool_kit.execute("read_file", {"path": path})
        assert text_read == content
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_unknown_tool() -> None:
    """Test for `execute` function with incorrect arguments."""

    with pytest.raises(ValueError):
        ToolKit([]).execute("dummy", {})


def test_tools_summary() -> None:
    """Test for the `get_summary` function."""

    summary = ToolKit([ReadFileTool(), WriteFileTool(), ShellTool()]).get_summary()
    for name in ["read_file", "write_file", "shell"]:
        assert f"Name: {name}\n" in summary
