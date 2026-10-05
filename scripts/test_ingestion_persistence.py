from pathlib import Path

from embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from embeddings.service import EmbeddingService
from ingestion.service import IngestionService
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


STORAGE_DIR = Path("storage/ingestion_test")


def main():
    print("=" * 60)
    print("DAY 11 - INGESTION PERSISTENCE TEST")
    print("=" * 60)

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_path = STORAGE_DIR / "vectors"
    keyword_path = STORAGE_DIR / "keywords.pkl"

    print("\n[1/7] Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding()

    embedding_service = EmbeddingService(
        embedding_model
    )

    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    print("\n[2/7] Creating indexes...")

    vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    keyword_store = BM25KeywordStore()

    print("Indexes created.")

    print("\n[3/7] Ingesting document...")

    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        keyword_store=keyword_store,
    )

    result = ingestion_service.ingest_file(
        "data/raw/sample.txt"
    )

    print(
        f"Indexed chunks: {result['chunks']}"
    )

    print("\n[4/7] Saving indexes...")

    vector_store.save(str(vector_path))
    keyword_store.save(str(keyword_path))

    print(f"FAISS saved to: {vector_path}")
    print(f"BM25 saved to: {keyword_path}")

    print("\n[5/7] Creating fresh index objects...")

    loaded_vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    loaded_keyword_store = BM25KeywordStore()

    loaded_vector_store.load(
        str(vector_path)
    )

    loaded_keyword_store.load(
        str(keyword_path)
    )

    print("Indexes loaded successfully.")

    print("\n[6/7] Searching loaded indexes...")

    query = "How does the system search documents?"

    query_vector = embedding_service.embed_text(
        query
    )

    vector_results = loaded_vector_store.search(
        query_vector=query_vector,
        top_k=3,
    )

    keyword_results = loaded_keyword_store.search(
        query=query,
        top_k=3,
    )

    print("\nFAISS TOP RESULT")
    print("-" * 60)
    print(vector_results[0].text)
    print(
        f"Score: {vector_results[0].score:.4f}"
    )

    print("\nBM25 TOP RESULT")
    print("-" * 60)
    print(keyword_results[0]["text"])
    print(
        f"Score: {keyword_results[0]['score']:.4f}"
    )

    print("\n[7/7] Persistence verification...")

    assert loaded_vector_store.index.ntotal > 0
    assert len(loaded_keyword_store.documents) > 0
    assert vector_results
    assert keyword_results

    print("FAISS vectors restored successfully.")
    print("BM25 documents restored successfully.")
    print("Search after reload works successfully.")

    print("\n" + "=" * 60)
    print("PERSISTENCE TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()