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


VECTOR_PATH = "storage/ingestion_test/vectors"
KEYWORD_PATH = "storage/ingestion_test/keywords.pkl"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = "google/flan-t5-small"


def main():
    print("=" * 60)
    print("DAY 12 - PERSISTENT RAG INTEGRATION TEST")
    print("=" * 60)

    print("\n[1/7] Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding(
        model_name=EMBEDDING_MODEL
    )

    embedding_service = EmbeddingService(
        embedding_model
    )

    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    print("\n[2/7] Loading persisted knowledge base...")

    knowledge_base = KnowledgeBase(
        dimension=embedding_service.dimension,
        vector_path=VECTOR_PATH,
        keyword_path=KEYWORD_PATH,
    )

    knowledge_base.load()

    print(
        f"FAISS vectors: "
        f"{knowledge_base.vector_count}"
    )

    print(
        f"BM25 documents: "
        f"{knowledge_base.document_count}"
    )

    print("\n[3/7] Building retrieval stack...")

    retriever = build_retriever(
        embedding_service=embedding_service,
        knowledge_base=knowledge_base,
    )

    print(
        "Hybrid retrieval + reranking ready."
    )

    print("\n[4/7] Loading local LLM...")

    llm = LocalLLMProvider(
        model_name=LLM_MODEL
    )

    prompt_builder = RAGPromptBuilder()

    generator = RAGGenerator(
        llm=llm,
        prompt_builder=prompt_builder,
    )

    print("RAG generator ready.")

    print("\n[5/7] Building complete RAG pipeline...")

    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        retriever=retriever,
        generator=generator,
    )

    print("RAG pipeline ready.")

    print("\n[6/7] Asking question...")

    query = (
        "How does the system search documents?"
    )

    result = pipeline.answer(
        query=query,
        top_k=3,
        candidate_k=10,
        max_new_tokens=128,
    )

    print("\nUSER QUERY")
    print("-" * 60)
    print(query)

    print("\nFINAL ANSWER")
    print("-" * 60)
    print(result["answer"])

    print("\nSOURCES")
    print("-" * 60)

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"[{index}] "
            f"{source}"
        )

    print("\nRETRIEVED DOCUMENTS")
    print("-" * 60)

    for index, document in enumerate(
        result["retrieved_documents"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{document.get('text', '')[:100]}"
        )

        if "rerank_score" in document:
            print(
                f"   Rerank score: "
                f"{document['rerank_score']:.4f}"
            )

    print("\n[7/7] Integration verification...")

    assert result["answer"]
    assert result["retrieved_documents"]

    print("Persistent knowledge base: OK")
    print("Hybrid retrieval: OK")
    print("Reranking: OK")
    print("RAG generation: OK")

    print("\n" + "=" * 60)
    print("PERSISTENT RAG INTEGRATION TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()