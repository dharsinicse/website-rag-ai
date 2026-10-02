from conversation.memory import ConversationMemory
from conversation.query_rewriter import QueryRewriter


class ConversationService:
    """Coordinate conversation memory and query rewriting."""

    def __init__(
        self,
        memory: ConversationMemory,
        query_rewriter: QueryRewriter,
    ):
        self.memory = memory
        self.query_rewriter = query_rewriter

    def prepare_query(self, query: str) -> str:
        history = self.memory.get_messages()

        return self.query_rewriter.rewrite(
            query=query,
            conversation_history=history,
        )

    def add_user_message(self, content: str) -> None:
        self.memory.add_message(
            role="user",
            content=content,
        )

    def add_assistant_message(self, content: str) -> None:
        self.memory.add_message(
            role="assistant",
            content=content,
        )

    def get_history(self) -> list[dict[str, str]]:
        return self.memory.get_messages()

    def clear(self) -> None:
        self.memory.clear()