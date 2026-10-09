
from config.settings import get_settings

from ingestion.processors.pipeline import DocumentProcessor
from ingestion.service import IngestionService

from api.rag_app import get_rag_pipeline


def get_ingestion_service() -> IngestionService:
    """Build ingestion service using the RAG pipeline's stores."""

    settings = get_settings()
    pipeline = get_rag_pipeline()

    hybrid_retriever = (
        pipeline.retriever.hybrid_retriever
    )

    return IngestionService(
        embedding_service=pipeline.embedding_service,
        vector_store=hybrid_retriever.vector_store,
        keyword_store=hybrid_retriever.keyword_store,
        document_processor=DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
    )