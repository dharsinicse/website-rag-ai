from dataclasses import dataclass


@dataclass
class ConversationMessage:
    role: str
    content: str


class ConversationMemory:
    """Store conversation messages for a single chat session."""

    def __init__(self, max_messages: int = 20):
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0")

        self.max_messages = max_messages
        self.messages: list[ConversationMessage] = []

    def add_message(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(
                "role must be 'user' or 'assistant'"
            )

        if not content.strip():
            raise ValueError("content cannot be empty")

        self.messages.append(
            ConversationMessage(
                role=role,
                content=content.strip(),
            )
        )

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> list[dict[str, str]]:
        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in self.messages
        ]

    def clear(self) -> None:
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)