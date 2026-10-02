from abc import ABC, abstractmethod


class QueryRewriter(ABC):
    """Abstract interface for conversational query rewriting."""

    @abstractmethod
    def rewrite(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        raise NotImplementedError


class SimpleQueryRewriter(QueryRewriter):
    """
    Basic query rewriter for the initial implementation.

    This version handles obvious conversational references without
    requiring an LLM. A model-based implementation can be added later.
    """

    FOLLOW_UP_PATTERNS = (
        "it",
        "its",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "how long",
        "how much",
        "what about",
        "when",
        "where",
        "why",
        "how",
    )

    def rewrite(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not conversation_history:
            return query.strip()

        normalized_query = query.strip().lower()

        is_follow_up = any(
            normalized_query.startswith(pattern)
            or f" {pattern} " in f" {normalized_query} "
            for pattern in self.FOLLOW_UP_PATTERNS
        )

        if not is_follow_up:
            return query.strip()

        previous_user_message = None

        for message in reversed(conversation_history):
            if message.get("role") == "user":
                previous_user_message = message.get("content", "").strip()
                break

        if not previous_user_message:
            return query.strip()

        return f"{previous_user_message} {query.strip()}"