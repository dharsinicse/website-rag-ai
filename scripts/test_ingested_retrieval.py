from embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from embeddings.service import EmbeddingService
from ingestion.service import IngestionService
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


def main():
    print("=" * 60)
    print("DAY 11 - INGESTED DOCUMENT RETRIEVAL TEST")
    print("=" * 60)

    print("\n[1/5] Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding()

    embedding_service = EmbeddingService(
        embedding_model
    )

    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    print("\n[2/5] Creating indexes...")

    vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    keyword_store = BM25KeywordStore()

    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        keyword_store=keyword_store,
    )

    print("Indexes ready.")

    print("\n[3/5] Ingesting sample document...")

    result = ingestion_service.ingest_file(
        "data/raw/sample.txt"
    )

    print(
        f"Indexed chunks: {result['chunks']}"
    )

    print("\n[4/5] Testing FAISS semantic retrieval...")

    query = "How does the system search documents?"

    query_vector = embedding_service.embed_text(
        query
    )

    vector_results = vector_store.search(
        query_vector=query_vector,
        top_k=3,
    )

    print("\nFAISS RESULTS")
    print("-" * 60)

    for index, result in enumerate(
        vector_results,
        start=1,
    ):
        print(
            f"{index}. "
            f"{result.text[:100]}"
        )
        print(
            f"   Score: {result.score:.4f}"
        )

    print("\n[5/5] Testing BM25 keyword retrieval...")

    keyword_results = keyword_store.search(
        query=query,
        top_k=3,
    )

    print("\nBM25 RESULTS")
    print("-" * 60)

    for index, result in enumerate(
        keyword_results,
        start=1,
    ):
        print(
            f"{index}. "
            f"{result['text'][:100]}"
        )
        print(
            f"   Score: "
            f"{result['score']:.4f}"
        )

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()