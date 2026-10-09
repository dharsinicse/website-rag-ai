from pathlib import Path

from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


class KnowledgeBase:
    """Load and manage the persisted RAG knowledge base."""

    def __init__(
        self,
        dimension: int,
        vector_path: str,
        keyword_path: str,
    ):
        self.vector_store = FAISSVectorStore(
            dimension=dimension
        )

        self.keyword_store = BM25KeywordStore()

        self.vector_path = Path(vector_path)
        self.keyword_path = Path(keyword_path)

    def load(self) -> None:
        """Load FAISS and BM25 indexes from disk."""

        if not self.vector_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{self.vector_path}"
            )

        if not self.keyword_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: "
                f"{self.keyword_path}"
            )

        self.vector_store.load(
            str(self.vector_path)
        )

        self.keyword_store.load(
            str(self.keyword_path)
        )

    
    def save(self) -> None:
        """Persist both vector and keyword indexes."""

        self.vector_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.keyword_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store.save(
            str(self.vector_path)
        )

        self.keyword_store.save(
            str(self.keyword_path)
        )

    @property
    def vector_count(self) -> int:
        """Return the number of indexed vectors."""

        return self.vector_store.index.ntotal

    @property
    def document_count(self) -> int:
        """Return the number of indexed keyword documents."""

        return len(self.keyword_store.documents)