from embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from embeddings.service import EmbeddingService
from retrieval.hybrid import HybridRetriever
from retrieval.knowledge_base import KnowledgeBase


VECTOR_PATH = "storage/ingestion_test/vectors"
KEYWORD_PATH = "storage/ingestion_test/keywords.pkl"


def main():
    print("=" * 60)
    print("DAY 12 - HYBRID RETRIEVAL DIAGNOSTIC")
    print("=" * 60)

    print("\n[1/4] Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding()

    embedding_service = EmbeddingService(
        embedding_model
    )

    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    print("\n[2/4] Loading knowledge base...")

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

    print("\n[3/4] Building hybrid retriever...")

    hybrid_retriever = HybridRetriever(
        vector_store=knowledge_base.vector_store,
        keyword_store=knowledge_base.keyword_store,
    )

    print("Hybrid retriever ready.")

    print("\n[4/4] Running query...")

    query = "How does the system search documents?"

    query_vector = embedding_service.embed_text(
        query
    )

    results = hybrid_retriever.search(
        query_vector=query_vector,
        query=query,
        top_k=5,
        candidate_k=5,
    )

    print("\nQUERY")
    print("-" * 60)
    print(query)

    print("\nHYBRID RESULTS BEFORE RERANKING")
    print("-" * 60)

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{index}. "
            f"{result.get('text', '')}"
        )

        print(
            f"   Source: "
            f"{result.get('source', '')}"
        )

        print(
            f"   Heading: "
            f"{result.get('heading', '')}"
        )

        print(
            f"   RRF score: "
            f"{result.get('rrf_score', 'N/A')}"
        )

        print(
            f"   Vector score: "
            f"{result.get('vector_score', 'N/A')}"
        )

        print(
            f"   Keyword score: "
            f"{result.get('keyword_score', 'N/A')}"
        )

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL DIAGNOSTIC COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()