from embeddings.service import EmbeddingService
from embeddings.sentence_transformer import SentenceTransformerEmbedding
from llm.generator import RAGGenerator
from llm.prompts import RAGPromptBuilder
from llm.providers.local import LocalLLMProvider
from retrieval.hybrid import HybridRetriever
from retrieval.knowledge_base import KnowledgeBase
from retrieval.pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("DAY 12 - PERSISTENT RAG WITHOUT RERANKER")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load embedding model
    # ---------------------------------------------------------
    print("\n[1/6] Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_service = EmbeddingService(
        embedding_model
    )

    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    # ---------------------------------------------------------
    # 2. Load persisted knowledge base
    # ---------------------------------------------------------
    print("\n[2/6] Loading knowledge base...")

    knowledge_base = KnowledgeBase(
        dimension=embedding_service.dimension,
        vector_path="storage/ingestion_test/vectors",
        keyword_path="storage/ingestion_test/keywords.pkl",
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

    # ---------------------------------------------------------
    # 3. Build HYBRID retriever
    #    IMPORTANT: NO RERANKER
    # ---------------------------------------------------------
    print("\n[3/6] Building hybrid retriever...")

    retriever = HybridRetriever(
        vector_store=knowledge_base.vector_store,
        keyword_store=knowledge_base.keyword_store,
    )

    print("Hybrid retrieval ready.")

    # ---------------------------------------------------------
    # 4. Load local LLM
    # ---------------------------------------------------------
    print("\n[4/6] Loading LLM...")

    llm = LocalLLMProvider(
        model_name="google/flan-t5-small"
    )

    prompt_builder = RAGPromptBuilder()

    generator = RAGGenerator(
        llm=llm,
        prompt_builder=prompt_builder,
    )

    print("RAG generator ready.")

    # ---------------------------------------------------------
    # 5. Build RAG pipeline
    # ---------------------------------------------------------
    print("\n[5/6] Building RAG pipeline...")

    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        retriever=retriever,
        generator=generator,
    )

    print("RAG pipeline ready.")

    # ---------------------------------------------------------
    # 6. Run real query
    # ---------------------------------------------------------
    print("\n[6/6] Running query...")

    query = "How does the system search documents?"

    result = pipeline.answer(
        query=query,
        top_k=3,
        candidate_k=10,
        max_new_tokens=128,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("USER QUERY")
    print("=" * 60)
    print(query)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(f"[{index}] {source}")

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for index, document in enumerate(
        result["retrieved_documents"],
        start=1,
    ):
        print(f"\n{index}. {document['text']}")

        print(
            f"   Heading: "
            f"{document.get('heading', '')}"
        )

        print(
            f"   Vector score: "
            f"{document.get('vector_score', 'N/A')}"
        )

        print(
            f"   Keyword score: "
            f"{document.get('keyword_score', 'N/A')}"
        )

        print(
            f"   RRF score: "
            f"{document.get('rrf_score', 'N/A')}"
        )

    print("\n" + "=" * 60)
    print("RERANKER: DISABLED")
    print("PERSISTENT KNOWLEDGE BASE: OK")
    print("HYBRID RETRIEVAL: OK")
    print("RAG GENERATION: OK")
    print("=" * 60)


if __name__ == "__main__":
    main()