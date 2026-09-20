"""Script to build the agent harness, which orchestrates all the parts."""

import json

from src.config import ChatHistory, config
from src.context_manager import ContextManager
from src.llm import LLM


class AgentHarness:
    """Class to build the agent harness."""

    def __init__(self, llm: LLM, context_manager: ContextManager) -> None:
        """Constructor of the class.

        Args:
            llm: LLM provider.
            context_manager: Context manager.
        """

        self.llm = llm
        self.context_manager = context_manager
        self.history: ChatHistory = []

    def _compact_history(self) -> None:
        """Replace conversation history with a summary after a successful LLM call."""

        request = [
            {"role": "system", "content": config.prompts.summarizer},
            {"role": "user", "content": json.dumps(self.history, ensure_ascii=False)},
        ]
        summary = self.llm.chat(request).strip()

        self.history = [
            {"role": "assistant", "content": f"Conversation summary:\n{summary}"}
        ]
        print("Chat history compacted.")

    def _handle_command(self, command: str) -> bool:
        """Handle a slash command; return False when the user requests exit.

        Args:
            command: Command executed by the user.

        Returns:
            `True` to continue with the agent loop, `False` otherwise.
        """

        match command:
            case "/help":
                print(
                    "/help   Show this help message.\n"
                    "/compact Summarize the chat history.\n"
                    "/reset  Clear the chat history.\n"
                    "/exit   Exit the agent."
                )
            case "/reset":
                self.history.clear()
                print("Chat history cleared.")
            case "/compact":
                self._compact_history()
            case "/exit":
                return False
            case _:
                print("Unknown command. Use /help for available commands.")

        return True

    def _handle_tool_call(self, response: str) -> bool:
        """Execute a tool request, recording its result or error in history.

        Args:
            response: Response of the LLM.

        Returns:
            `True` if the LLM calls a tool, `False` otherwise.
        """

        try:
            call = json.loads(response)
        except json.JSONDecodeError:
            return False

        if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
            return False

        name, arguments = call["tool"], call["arguments"]
        result = self.context_manager.tool_kit.execute(name, arguments)
        self.history.append(
            {"role": "user", "content": f"Tool result ({name}):\n{result}"}
        )

        return True

    def _process_turn(self, query: str) -> None:
        """Processes one turn of the running agent.

        Args:
            query: Query of the user.
        """

        self.history.append(
            {
                "role": "user",
                "content": self.context_manager.get_prefix() + "\n\n" + query,
            }
        )

        limit = config.tools.max_calls_per_turn
        for _ in range(limit):
            response = self.llm.chat(self.history)
            self.history.append({"role": "assistant", "content": response})
            if not self._handle_tool_call(response):
                return

        self.history.append(
            {
                "role": "assistant",
                "content": f"Stopped: tool-call limit reached ({limit}).",
            }
        )

    def run(self) -> None:
        """Runs the agent harness."""

        while True:
            # Query of the user
            query = input("User query: ").strip()

            # Commands
            if query.startswith("/"):
                if not self._handle_command(query):
                    break
                continue

            # Agent response
            self._process_turn(query)
            print(f"Agent: {self.history[-1]["content"]}")
