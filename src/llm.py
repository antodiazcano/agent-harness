"""Script to define the LLM used."""

from dotenv import load_dotenv
from groq import Groq

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

    def chat(self, query: str, history: ChatHistory) -> str:
        """Uses the LLM to response the user query.

        Args:
            query: Query of the user.
            history: Chat history.

        Returns:
            LLM answer.

        Raises:
            RuntimeError: If there's an error while executing the model.
        """

        response = (
            self.client.chat.completions.create(messages=history, model=self.model)
            .choices[0]
            .message.content
        )

        if response is None:
            raise RuntimeError("There was an error while processing the LLM response!")

        return response
