from embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from embeddings.service import EmbeddingService
from ingestion.processors.pipeline import (
    DocumentProcessor,
)
from retrieval.vector.faiss_store import (
    FAISSVectorStore,
)


INDEX_PATH = (
    "storage/vector_store/"
    "test_index.faiss"
)


def main():
    print("Loading embedding model...")

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    embedding_service = EmbeddingService(
        embedding_model
    )

    print()
    print("Creating fresh FAISS store...")

    store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    print("Loading persisted index...")

    store.load(INDEX_PATH)

    print(
        f"Loaded vectors: "
        f"{store.index.ntotal}"
    )

    query = (
        "Can users upload PDF documents?"
    )

    query_embedding = (
        embedding_model.embed_text(query)
    )

    results = store.search(
        query_vector=query_embedding,
        top_k=3,
    )

    print()
    print("=" * 60)
    print("RELOADED FAISS SEARCH")
    print("=" * 60)

    print(f"Query: {query}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"Rank: {rank}")
        print(
            f"Score: {result.score:.4f}"
        )
        print(
            f"Heading: {result.heading}"
        )
        print(
            f"Source: {result.source}"
        )
        print(
            f"Text: {result.text}"
        )

        print("-" * 60)

    assert results

    assert (
        results[0].heading
        == "Documents"
    )

    print()
    print("=" * 60)
    print("RELOADED FAISS SEARCH TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()