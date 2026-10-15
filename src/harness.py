"""Script to build the agent harness, which orchestrates all the parts."""

import json
import sys

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
        self.tools = context_manager.tool_kit
        self.history: ChatHistory = []

    @staticmethod
    def _style(text: str, code: str) -> str:
        """Add ANSI styling when writing to an interactive terminal.

        Args:
            text: Text to display.
            code: ANSI style code to apply.

        Returns:
            Styled text in an interactive terminal, otherwise the original text.
        """

        return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

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
                    f"{self._style('Commands', '1;36')}\n"
                    "  /help      Show this help message.\n"
                    "  /compact   Summarize the chat history.\n"
                    "  /reset     Clear the chat history.\n"
                    "  /exit      Exit the agent."
                )
            case "/reset":
                self.history.clear()
                print(f"{self._style('✓', '32')} Chat history cleared.")
            case "/compact":
                request = [
                    {"role": "system", "content": config.prompts.summarizer},
                    {
                        "role": "user",
                        "content": json.dumps(self.history, ensure_ascii=False),
                    },
                ]
                summary = self.llm.chat(request).strip()
                self.history = [
                    {
                        "role": "assistant",
                        "content": f"Conversation summary:\n{summary}",
                    }
                ]
                print(f"{self._style('✓', '32')} Chat history compacted.")
            case "/exit":
                return False
            case _:
                print(
                    f"{self._style('!', '33')} Unknown command. "
                    "Use /help for available commands."
                )

        return True

    def _handle_delegation(self, response: str) -> bool:
        """Run a delegation request in a fresh child and record its final answer.

        Args:
            response: Response of the LLM.

        Returns:
            `True` if the LLM delegates into a subagent, `False` otherwise.
        """

        try:
            call = json.loads(response)
        except json.JSONDecodeError:
            return False

        if not isinstance(call, dict) or set(call) != {"delegate"}:
            return False

        print(self._style("◇ Running subagent...", "33"))

        child = AgentHarness(self.llm, self.context_manager)
        child._process_turn(call["delegate"])

        result = child.history[-1]["content"]
        self.history.append({"role": "user", "content": f"Subagent result:\n{result}"})

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
        print(
            f"{self._style(f'◇ Using tool: {name}', '1;33')}\n"
            f"  {self._style(str(arguments), '2')}"
        )
        result = self.tools.execute(name, arguments)
        self.history.append(
            {"role": "user", "content": f"Tool result ({name}):\n{result}"}
        )

        return True

    def _process_turn(self, query: str) -> None:
        """Processes one turn of the running agent.

        Args:
            query: Query of the user.
        """

        prefix = self.context_manager.get_prefix()
        self.history.append({"role": "user", "content": query})

        limit = config.tools.max_calls_per_turn
        for _ in range(limit):
            messages = [{"role": "system", "content": prefix}, *self.history]
            response = self.llm.chat(messages)
            self.history.append({"role": "assistant", "content": response})
            if not (
                self._handle_delegation(response) or self._handle_tool_call(response)
            ):
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
            query = input(f"\n{self._style('You >', '1;36')} ").strip()

            # Commands
            if query.startswith("/"):
                if not self._handle_command(query):
                    break
                continue

            # Agent response
            self._process_turn(query)
            print(f"\n{self._style('Tony >', '1;32')} {self.history[-1]["content"]}")
