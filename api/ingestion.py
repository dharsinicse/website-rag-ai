
from config.settings import get_settings

from embeddings.service import EmbeddingService
from ingestion.processors.pipeline import DocumentProcessor
from ingestion.service import IngestionService

from api.rag_app import get_rag_pipeline


def get_ingestion_service() -> IngestionService:
    """Build an ingestion service using the shared RAG indexes."""

    settings = get_settings()
    pipeline = get_rag_pipeline()

    # The retriever uses these same store instances.
    knowledge_base = pipeline.retriever.hybrid_retriever

    embedding_service = pipeline.embedding_service

    processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return IngestionService(
        embedding_service=embedding_service,
        vector_store=knowledge_base.vector_store,
        keyword_store=knowledge_base.keyword_store,
        document_processor=processor,
    )