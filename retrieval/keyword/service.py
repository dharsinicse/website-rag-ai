from retrieval.keyword.bm25_store import BM25KeywordStore
from ingestion.chunking.text_chunker import TextChunk


class KeywordRetrievalService:
    """Build and query a BM25 index from RAG text chunks."""

    def __init__(self, store: BM25KeywordStore):
        self.store = store

    def index_chunks(self, chunks: list[TextChunk]) -> None:
        """Add text chunks to the BM25 index."""

        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]

        metadata = [
            {
                "source": chunk.source,
                "title": chunk.title,
                "heading": chunk.heading,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        self.store.add(
            texts=texts,
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search indexed chunks."""

        return self.store.search(
            query=query,
            top_k=top_k,
        )