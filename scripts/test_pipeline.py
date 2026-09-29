from ingestion.processors.pipeline import (
    DocumentProcessor,
)


processor = DocumentProcessor(
    chunk_size=100,
    chunk_overlap=20,
)


raw_text = """
# Product Overview

Our platform provides enterprise RAG capabilities.

It supports website ingestion, document processing,
semantic search, keyword search, reranking, and
grounded answer generation.

# Features

The system is designed for scalable AI applications.
"""


result = processor.process(
    text=raw_text,
    source="https://example.com/docs",
    title="Product Documentation",
    metadata={
        "document_type": "website",
        "language": "en",
    },
)


print(f"Source: {result.source}")
print(f"Title: {result.title}")
print(f"Chunks created: {len(result.chunks)}")
print()

for chunk in result.chunks:
    print("=" * 60)
    print(f"Chunk index: {chunk.chunk_index}")
    print(f"Heading: {chunk.heading}")
    print(f"Length: {len(chunk.text)}")
    print(f"Text: {chunk.text}")
    print(f"Metadata: {chunk.metadata}")