"""Script to define the LLM used."""

from typing import cast

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam

from src.config import ChatHistory


class LLM:
    """Class to implement the Groq model."""

    def __init__(self, model: str, temperature: float = 1.0) -> None:
        """Constructor of the class.

        Args:
            model: Model used.
            temperature: Temperature of the LLM.
        """

        load_dotenv()

        self.client = Groq()
        self.model = model
        self.temperature = temperature

    def chat(self, history: ChatHistory) -> str:
        """Uses the LLM to response the user query.

        Args:
            history: Chat history.

        Returns:
            Assistant's text response.

        Raises:
            RuntimeError: If there's an error while executing the model.
        """

        response = (
            self.client.chat.completions.create(
                messages=cast(list[ChatCompletionMessageParam], history),
                model=self.model,
            )
            .choices[0]
            .message.content
        )

        if response is None:
            raise RuntimeError("There was an error while processing the LLM response!")

        return response
