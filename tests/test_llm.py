"""Script to test the LLM."""

from unittest.mock import Mock, patch

import pytest

from src.config import ChatHistory, config
from src.llm import LLM


def test_chat_normal() -> None:
    """Test the LLM chat function."""

    with patch("src.llm.Groq") as groq:
        groq.return_value.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Mock response"))]
        )
        llm = LLM(config.llm.groq_model)

        history: ChatHistory = [
            {"role": "system", "content": "This is a dummy prompt."}
        ]
        response = llm.chat(history)

        assert response == "Mock response", "Incorrect response!"
        groq.return_value.chat.completions.create.assert_called_once_with(
            messages=history, model=config.llm.groq_model
        )


def test_chat_raises_error_when_content_is_none() -> None:
    """Test of the LLM chat function when the response has no content."""

    with patch("src.llm.Groq") as groq:
        groq.return_value.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=None))]
        )
        llm = LLM(config.llm.groq_model)
        history: ChatHistory = [
            {"role": "system", "content": "This is a dummy prompt."}
        ]

        with pytest.raises(RuntimeError):
            llm.chat(history)
