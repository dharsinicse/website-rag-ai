from sentence_transformers import SentenceTransformer

from embeddings.base import EmbeddingModel


class SentenceTransformerEmbedding(
    EmbeddingModel
):
    """Sentence Transformer based embedding model."""

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for one text."""

        if not text.strip():
            raise ValueError(
                "Text cannot be empty"
            )

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""

        return self.model.get_embedding_dimension()