from pathlib import Path
from typing import Any

from embeddings.service import EmbeddingService
from ingestion.loaders.document import (
    load_docx,
    load_html,
    load_pdf,
    load_text,
)
from ingestion.processors.pipeline import DocumentProcessor
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


class IngestionService:
    """Load, process, embed, and index documents."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        keyword_store: BM25KeywordStore,
        document_processor: DocumentProcessor | None = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.keyword_store = keyword_store

        self.document_processor = (
            document_processor
            or DocumentProcessor()
        )

    def load_file(self, file_path: str) -> str:
        """Load supported file types into plain text."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return load_pdf(file_path)

        if suffix == ".docx":
            return load_docx(file_path)

        if suffix in {".txt", ".md"}:
            return load_text(file_path)

        if suffix in {".html", ".htm"}:
            return load_html(file_path)

        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    def ingest_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest one document into FAISS and BM25."""

        path = Path(file_path)

        text = self.load_file(file_path)

        document_metadata = {
            "document_type": path.suffix.lower().lstrip("."),
            **(metadata or {}),
        }

        processed = self.document_processor.process(
            text=text,
            source=str(path),
            title=path.stem,
            metadata=document_metadata,
        )

        chunks = processed.chunks

        if not chunks:
            return {
                "source": str(path),
                "title": path.stem,
                "chunks": 0,
                "status": "empty",
            }

        embeddings = (
            self.embedding_service.embed_chunks(chunks)
        )

        chunk_metadata = []

        for chunk in chunks:
            chunk_metadata.append(
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.title,
                    "heading": chunk.heading,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                }
            )

        self.vector_store.add(
            vectors=embeddings,
            metadata=chunk_metadata,
        )

        self.keyword_store.add(
            texts=[chunk.text for chunk in chunks],
            metadata=chunk_metadata,
        )

        return {
            "source": str(path),
            "title": path.stem,
            "chunks": len(chunks),
            "status": "indexed",
        }