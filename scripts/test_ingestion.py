from embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from embeddings.service import EmbeddingService
from ingestion.service import IngestionService
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


def main():
    print("=" * 60)
    print("DAY 11 - DOCUMENT INGESTION TEST")
    print("=" * 60)

    print("\n[1/6] Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding()

    embedding_service = EmbeddingService(
        embedding_model
    )

    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    print("\n[2/6] Creating vector store...")

    vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    print("FAISS vector store ready.")

    print("\n[3/6] Creating keyword store...")

    keyword_store = BM25KeywordStore()

    print("BM25 keyword store ready.")

    print("\n[4/6] Creating ingestion service...")

    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        keyword_store=keyword_store,
    )

    print("Ingestion service ready.")

    print("\n[5/6] Ingesting sample document...")

    result = ingestion_service.ingest_file(
        "data/raw/sample.txt"
    )

    print("\nINGESTION RESULT")
    print("-" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n[6/6] Checking index sizes...")

    print(
        f"FAISS vectors: "
        f"{vector_store.index.ntotal}"
    )

    print(
        f"BM25 documents: "
        f"{len(keyword_store.documents)}"
    )

    print("\n" + "=" * 60)
    print("INGESTION TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()