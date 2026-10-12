"""Script to define the LLM used."""

import json
from typing import cast

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from src.config import ChatHistory
from src.tools import Tool


class LLM:
    """Class to implement the Groq model."""

    def __init__(
        self, model: str, temperature: float = 1.0, tools: list[Tool] | None = None
    ) -> None:
        """Constructor of the class.

        Args:
            model: Model used.
            temperature: Temperature of the LLM.
            tools: Tools available to the model.
        """

        load_dotenv()

        self.client = Groq()
        self.model = model
        self.temperature = temperature
        self.tools = self._init_tools(tools)

    @staticmethod
    def _init_tools(tools: list[Tool] | None) -> list[ChatCompletionToolParam]:
        """Build the Groq schemas for tools and delegation.

        Args:
            tools: Tools available to the model.

        Returns:
            Groq tool schemas.
        """

        schemas: list[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            name: {"type": "string", "description": description}
                            for name, description in tool.args.items()
                        },
                        "required": list(tool.args),
                    },
                },
            }
            for tool in tools or []
        ]
        if tools is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "delegate",
                        "description": "Delegate a task to a subagent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "delegate": {
                                    "type": "string",
                                    "description": (
                                        "Self-contained task and relevant context."
                                    ),
                                }
                            },
                            "required": ["delegate"],
                        },
                    },
                }
            )

        return schemas

    def chat(self, history: ChatHistory) -> str:
        """Uses the LLM to response the user query.

        Args:
            history: Chat history.

        Returns:
            Assistant's text response.

        Raises:
            RuntimeError: If there's an error while executing the model.
        """

        message = (
            self.client.chat.completions.create(
                messages=cast(list[ChatCompletionMessageParam], history),
                model=self.model,
                temperature=self.temperature,
                tools=self.tools or None,
                parallel_tool_calls=False,
            )
            .choices[0]
            .message
        )

        if message.tool_calls:
            function = message.tool_calls[0].function
            arguments = json.loads(function.arguments)
            if function.name == "delegate":
                return json.dumps(arguments)
            return json.dumps({"tool": function.name, "arguments": arguments})

        if message.content is None:
            raise RuntimeError("There was an error while processing the LLM response!")

        return message.content
