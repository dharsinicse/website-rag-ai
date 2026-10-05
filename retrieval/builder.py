from embeddings.service import EmbeddingService
from retrieval.hybrid import (
    HybridRetriever,
    RerankedHybridRetriever,
)
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker


def build_retriever(
    embedding_service: EmbeddingService,
    knowledge_base: KnowledgeBase,
) -> RerankedHybridRetriever:
    """Build the hybrid retrieval + reranking stack."""

    hybrid_retriever = HybridRetriever(
        vector_store=knowledge_base.vector_store,
        keyword_store=knowledge_base.keyword_store,
    )

    reranker = CrossEncoderReranker()

    return RerankedHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )