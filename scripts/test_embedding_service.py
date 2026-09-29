from embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from embeddings.service import EmbeddingService
from ingestion.processors.pipeline import (
    DocumentProcessor,
)


def main():
    print("Creating document processor...")

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
    """

    document = processor.process(
        text=text,
        source="https://example.com/docs",
        title="Product Documentation",
        metadata={
            "document_type": "website",
        },
    )

    print(
        f"Chunks created: "
        f"{len(document.chunks)}"
    )

    print()
    print("Loading embedding model...")

    embedding_model = (
        SentenceTransformerEmbedding()
    )

    service = EmbeddingService(
        embedding_model
    )

    embeddings = service.embed_chunks(
        document.chunks
    )

    print()
    print("=" * 60)
    print("EMBEDDING SERVICE TEST")
    print("=" * 60)

    print(
        f"Chunks: {len(document.chunks)}"
    )

    print(
        f"Embeddings: {len(embeddings)}"
    )

    print(
        f"Dimension: {service.dimension}"
    )

    print()

    for index, (
        chunk,
        embedding,
    ) in enumerate(
        zip(
            document.chunks,
            embeddings,
        )
    ):
        print("-" * 60)
        print(f"Chunk index: {chunk.chunk_index}")
        print(f"Heading: {chunk.heading}")
        print(f"Source: {chunk.source}")
        print(f"Vector length: {len(embedding)}")

    assert len(embeddings) == len(
        document.chunks
    )

    assert all(
        len(embedding) == service.dimension
        for embedding in embeddings
    )

    print()
    print("=" * 60)
    print("EMBEDDING SERVICE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()