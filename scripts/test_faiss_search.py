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


def main():
    processor = DocumentProcessor(
        chunk_size=150,
        chunk_overlap=30,
    )

    text = """
    # Product Overview

    Our platform provides enterprise RAG capabilities.

    The system supports semantic search, keyword search,
    document ingestion, and grounded answer generation.

    # Security

    The platform validates uploaded documents and protects
    the system against unsafe inputs.

    # Documents

    Users can upload PDF and DOCX documents to the platform.
    """

    document = processor.process(
        text=text,
        source="https://example.com/docs",
        title="Product Documentation",
        metadata={
            "document_type": "website",
        },
    )

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    embedding_service = EmbeddingService(
        embedding_model
    )

    embeddings = (
        embedding_service.embed_chunks(
            document.chunks
        )
    )

    metadata = [
        {
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "heading": chunk.heading,
            "source": chunk.source,
            "title": chunk.title,
            "page": chunk.page,
            "metadata": chunk.metadata,
        }
        for chunk in document.chunks
    ]

    vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    vector_store.add(
        vectors=embeddings,
        metadata=metadata,
    )

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    query = (
        "Can users upload PDF documents?"
    )

    print("=" * 60)
    print("FAISS SEMANTIC SEARCH")
    print("=" * 60)

    print(f"Query: {query}")

    query_embedding = (
        embedding_model.embed_text(query)
    )

    results = vector_store.search(
        query_vector=query_embedding,
        top_k=3,
    )

    print()
    print("SEARCH RESULTS")
    print("=" * 60)

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

    print()
    print("FAISS SEARCH TEST PASSED")


if __name__ == "__main__":
    main()