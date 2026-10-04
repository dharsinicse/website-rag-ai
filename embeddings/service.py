from embeddings.base import EmbeddingModel
from ingestion.chunking.text_chunker import TextChunk


class EmbeddingService:
    """Provide embedding operations for documents and queries."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        return self.embedding_model.embed_text(text)

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        return self.embedding_model.embed_texts(texts)

    def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[list[float]]:
        """Generate embeddings for document chunks."""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        return self.embedding_model.embed_texts(texts)

    @property
    def dimension(self) -> int:
        return self.embedding_model.dimension