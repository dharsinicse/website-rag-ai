from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Base interface for vector stores."""

    @abstractmethod
    def add(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add vectors and their metadata."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for the most similar vectors."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the vector store."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store."""
        raise NotImplementedError