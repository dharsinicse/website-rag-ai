from embeddings.sentence_transformer import SentenceTransformerEmbedding
from embeddings.service import EmbeddingService
from retrieval.hybrid import HybridRetriever
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker


VECTOR_PATH = "storage/ingestion_test/vectors"
KEYWORD_PATH = "storage/ingestion_test/keywords.pkl"


def main():
    print("=" * 60)
    print("DAY 12 - RERANKER DIAGNOSTIC")
    print("=" * 60)

    print("\n[1/5] Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding()
    embedding_service = EmbeddingService(embedding_model)

    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    print("\n[2/5] Loading knowledge base...")

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

    print("\n[3/5] Running hybrid retrieval...")

    hybrid_retriever = HybridRetriever(
        vector_store=knowledge_base.vector_store,
        keyword_store=knowledge_base.keyword_store,
    )

    query = "How does the system search documents?"

    query_vector = embedding_service.embed_text(query)

    results = hybrid_retriever.search(
        query_vector=query_vector,
        query=query,
        top_k=5,
        candidate_k=5,
    )

    print("Hybrid retrieval completed.")

    print("\n[4/5] Loading Cross-Encoder reranker...")

    reranker = CrossEncoderReranker()

    print(
        f"Reranker model: "
        f"{reranker.model_name}"
    )

    print("\n[5/5] Reranking results...")

    reranked_results = reranker.rerank(
        query=query,
        results=results,
        top_k=5,
    )

    print("\nQUERY")
    print("-" * 60)
    print(query)

    print("\nRESULTS AFTER RERANKING")
    print("-" * 60)

    for index, result in enumerate(
        reranked_results,
        start=1,
    ):
        print(f"\n{index}. {result.get('text', '')}")
        print(
            f"   Heading: "
            f"{result.get('heading', '')}"
        )
        print(
            f"   Vector score: "
            f"{result.get('vector_score', 'N/A')}"
        )
        print(
            f"   Keyword score: "
            f"{result.get('keyword_score', 'N/A')}"
        )
        print(
            f"   Rerank score: "
            f"{result.get('rerank_score', 'N/A')}"
        )

    print("\n" + "=" * 60)
    print("RERANKER DIAGNOSTIC COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()