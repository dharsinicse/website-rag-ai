from ingestion.chunking.text_chunker import TextChunker
from ingestion.processors.sections import (
    DocumentSection,
)


section = DocumentSection(
    heading="Product Overview",
    text=(
        "Our platform provides enterprise RAG capabilities. "
        "It supports website ingestion, document processing, "
        "semantic search, keyword search, reranking, and "
        "grounded answer generation. "
        "The system is designed for scalable AI applications."
    ),
)


chunker = TextChunker(
    chunk_size=100,
    chunk_overlap=20,
)

chunks = chunker.chunk_sections(
    [section],
    source="https://example.com/docs",
    title="Product Documentation",
    metadata={
        "document_type": "website",
        "language": "en",
    },
)

print(f"Chunks created: {len(chunks)}")
print()

for chunk in chunks:
    print("=" * 60)
    print(f"Chunk index: {chunk.chunk_index}")
    print(f"Heading: {chunk.heading}")
    print(f"Source: {chunk.source}")
    print(f"Title: {chunk.title}")
    print(f"Page: {chunk.page}")
    print(f"Metadata: {chunk.metadata}")
    print(f"Length: {len(chunk.text)}")
    print(f"Text: {chunk.text}")