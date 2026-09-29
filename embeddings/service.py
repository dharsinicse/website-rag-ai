from embeddings.base import EmbeddingModel
from ingestion.chunking.text_chunker import TextChunk


class EmbeddingService:
    """Generate embeddings for RAG text chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
    ):
        self.embedding_model = embedding_model

    def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[list[float]]:
        """Generate embeddings for all chunks."""

        if not chunks:
            return []

        texts = [
            chunk.text
            for chunk in chunks
        ]

        return self.embedding_model.embed_texts(
            texts
        )

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""

        return self.embedding_model.dimension