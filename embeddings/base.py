from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """Base interface for embedding models."""

    @abstractmethod
    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for one text."""
        raise NotImplementedError

    @abstractmethod
    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        raise NotImplementedError