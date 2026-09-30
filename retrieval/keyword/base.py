from abc import ABC, abstractmethod
from typing import Any


class KeywordStore(ABC):
    """Abstract interface for keyword-based retrieval."""

    @abstractmethod
    def add(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add documents to the keyword index."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search documents using keyword matching."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the keyword index."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a persisted keyword index."""
        raise NotImplementedError