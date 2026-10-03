from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        raise NotImplementedError