"""Script to build the agent harness, which orchestrates all the parts."""

from src.config import ChatHistory
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

    def _process_turn(self, query: str) -> None:
        """Processes one turn of the running agent.

        Args:
            query: Query of the user.
        """

        self.history.append(
            {"role": "user", "content": self.context_manager.get_prefix() + query}
        )
        response = self.llm.chat(query, self.history)
        self.history.append({"role": "assistant", "content": response})

    def run(self) -> None:
        """Runs the agent harness.

        Returns:
            LLM answer.
        """

        while True:
            query = input("User query (`q` to quit): ")
            if query == "q":
                break
            self._process_turn(query)
            print(f"Agent: {self.history[-1]["content"]}")
