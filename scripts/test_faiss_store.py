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

    print(
        f"Chunks created: {len(document.chunks)}"
    )

    print()
    print("Loading embedding model...")

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

    print(
        f"Embeddings created: {len(embeddings)}"
    )

    metadata = []

    for chunk in document.chunks:
        metadata.append(
            {
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "heading": chunk.heading,
                "source": chunk.source,
                "title": chunk.title,
                "page": chunk.page,
                "metadata": chunk.metadata,
            }
        )

    print()
    print("Creating FAISS vector store...")

    vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    vector_store.add(
        vectors=embeddings,
        metadata=metadata,
    )

    print()
    print("=" * 60)
    print("FAISS STORE TEST")
    print("=" * 60)

    print(
        f"Vectors stored: "
        f"{vector_store.index.ntotal}"
    )

    print(
        f"Metadata stored: "
        f"{len(vector_store.metadata)}"
    )

    assert (
        vector_store.index.ntotal
        == len(document.chunks)
    )

    assert (
        len(vector_store.metadata)
        == len(document.chunks)
    )

    print()
    print(
        "FAISS STORE TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()