from config.settings import get_settings

from embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from embeddings.service import EmbeddingService

from llm.generator import RAGGenerator
from llm.prompts import RAGPromptBuilder
from llm.providers.local import LocalLLMProvider

from retrieval.builder import build_retriever
from retrieval.knowledge_base import KnowledgeBase
from retrieval.pipeline import RAGPipeline


def build_rag_pipeline() -> RAGPipeline:
    """Build the complete RAG pipeline from persisted indexes."""

    settings = get_settings()

    # 1. Embedding model
    embedding_model = SentenceTransformerEmbedding(
        model_name=settings.embedding_model,
    )

    embedding_service = EmbeddingService(
        embedding_model=embedding_model,
    )

    # 2. Load persisted knowledge base
    knowledge_base = KnowledgeBase(
        dimension=embedding_service.dimension,
        vector_path=settings.vector_index_path,
        keyword_path=settings.keyword_index_path,
    )

    knowledge_base.load()

    # 3. Hybrid retrieval + reranking
    retriever = build_retriever(
        embedding_service=embedding_service,
        knowledge_base=knowledge_base,
    )

    # 4. Local LLM
    llm = LocalLLMProvider(
        model_name="google/flan-t5-small",
    )

    # 5. Prompt builder + generator
    prompt_builder = RAGPromptBuilder()

    generator = RAGGenerator(
        llm=llm,
        prompt_builder=prompt_builder,
    )

    # 6. Complete RAG pipeline
    return RAGPipeline(
        embedding_service=embedding_service,
        retriever=retriever,
        generator=generator,
    )

_cached_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Return the cached RAG pipeline."""

    global _cached_pipeline

    if _cached_pipeline is None:
        _cached_pipeline = build_rag_pipeline()

    return _cached_pipeline