import hashlib
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
from retrieval.knowledge_base_persistence import save_staged_indexes


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

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate a SHA-256 fingerprint for a file."""
        sha256 = hashlib.sha256()
        with Path(file_path).open("rb") as file:
            for block in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):sha256.update(block)

        return sha256.hexdigest()

    def ingest_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest one document into FAISS and BM25."""
        path = Path(file_path)

        file_hash = self.calculate_file_hash(file_path)
        for item in self.keyword_store.metadata:
            existing_metadata = item.get("metadata", {})
            if existing_metadata.get("file_hash") == file_hash:
                return {
                    "source": str(path),
                    "title": path.stem,
                    "chunks": 0,
                    "status": "duplicate",
                }
        text = self.load_file(file_path)

        document_metadata = {
            **(metadata or {}),
            "document_type": path.suffix.lower().lstrip("."),
            "file_hash": file_hash,
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

    def delete_document(
        self,
        source: str,
        vector_path: str,
        keyword_path: str,
    ) -> dict[str, Any]:
        """Delete a source from both indexes and persist the result."""
        if not source:
            raise ValueError("Source cannot be empty")

        vector_store = self.vector_store
        keyword_store = self.keyword_store

        vector_count = sum(
            item.get("source") == source
            for item in vector_store.metadata
        )
        keyword_count = sum(
            item.get("source") == source
            for item in keyword_store.metadata
        )

        if vector_count != keyword_count:
            raise RuntimeError(
                "Vector and keyword indexes are inconsistent; "
                "deletion was not performed."
            )

        if vector_count == 0:
            return {
                "source": source,
                "deleted_chunks": 0,
                "status": "not_found",
            }

        # Keep in-memory copies so a failure can restore the live stores.
        original_vectors = vector_store.index
        original_vector_metadata = list(vector_store.metadata)
        original_documents = list(keyword_store.documents)
        original_keyword_metadata = list(keyword_store.metadata)
        original_tokens = list(keyword_store.tokenized_documents)
        original_bm25 = keyword_store.bm25

        try:
            vector_deleted = vector_store.delete_by_source(source)
            keyword_deleted = keyword_store.delete_by_source(source)

            if vector_deleted != keyword_deleted:
                raise RuntimeError(
                    "Vector and keyword deletion counts do not match."
                )

            save_staged_indexes(
                vector_store=vector_store,
                keyword_store=keyword_store,
                vector_path=vector_path,
                keyword_path=keyword_path,
            )
        except Exception:
            vector_store.index = original_vectors
            vector_store.metadata = original_vector_metadata
            keyword_store.documents = original_documents
            keyword_store.metadata = original_keyword_metadata
            keyword_store.tokenized_documents = original_tokens
            keyword_store.bm25 = original_bm25
            raise

        return {
            "source": source,
            "deleted_chunks": vector_deleted,
            "status": "deleted",
        }