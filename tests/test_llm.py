"""Script to test the LLM."""

from src.config import ChatHistory, config
from src.llm import LLM


def test_chat() -> None:
    """Script to test the LLM chat function."""

    llm = LLM(config.llm.groq_model)

    query = ""
    history: ChatHistory = [{"role": "system", "content": "This is a dummy prompt."}]

    response = llm.chat(query, history)

    assert isinstance(response, str), "Incorrect response!"
