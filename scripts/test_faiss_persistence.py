from pathlib import Path

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


def build_store():
    processor = DocumentProcessor(
        chunk_size=150,
        chunk_overlap=30,
    )

    text = """
    # Product Overview

    Our platform provides enterprise RAG capabilities.

    # Documents

    Users can upload PDF and DOCX documents
    to the platform.

    # Security

    The platform validates uploaded documents
    and protects the system against unsafe inputs.
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

    store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    store.add(
        vectors=embeddings,
        metadata=metadata,
    )

    return store, embedding_model


def main():
    print("Building FAISS store...")

    store, embedding_model = build_store()

    original_count = store.index.ntotal

    print(
        f"Original vectors: {original_count}"
    )

    print()
    print("Saving FAISS index...")

    store.save(INDEX_PATH)

    print(
        f"Index saved to: {INDEX_PATH}"
    )

    metadata_path = Path(
        INDEX_PATH
    ).with_suffix(
        ".metadata.npy"
    )

    assert Path(INDEX_PATH).exists()
    assert metadata_path.exists()

    print(
        f"Metadata saved to: {metadata_path}"
    )

    print()
    print("Creating new FAISS store...")

    loaded_store = FAISSVectorStore(
        dimension=embedding_model.dimension
    )

    loaded_store.load(INDEX_PATH)

    print(
        f"Loaded vectors: "
        f"{loaded_store.index.ntotal}"
    )

    print(
        f"Loaded metadata: "
        f"{len(loaded_store.metadata)}"
    )

    assert (
        loaded_store.index.ntotal
        == original_count
    )

    assert (
        len(loaded_store.metadata)
        == original_count
    )

    print()
    print("=" * 60)
    print("FAISS PERSISTENCE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()