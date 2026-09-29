from pathlib import Path

from ingestion.loaders.document import load_text
from ingestion.processors.pipeline import DocumentProcessor


TEST_FILE = Path(
    "data/test_documents/sample.txt"
)


def main():
    print("=" * 60)
    print("DOCUMENT INGESTION INTEGRATION TEST")
    print("=" * 60)

    # 1. Load document
    text = load_text(
        str(TEST_FILE)
    )

    print(f"Loaded file: {TEST_FILE}")
    print(f"Characters: {len(text)}")
    print()

    # 2. Process document
    processor = DocumentProcessor(
        chunk_size=150,
        chunk_overlap=30,
    )

    result = processor.process(
        text=text,
        source=str(TEST_FILE),
        title=TEST_FILE.stem,
        metadata={
            "document_type": "txt",
            "test_document": True,
        },
    )

    # 3. Display results
    print(f"Chunks created: {len(result.chunks)}")
    print()

    for chunk in result.chunks:
        print("-" * 60)
        print(f"Chunk index : {chunk.chunk_index}")
        print(f"Heading     : {chunk.heading}")
        print(f"Source      : {chunk.source}")
        print(f"Title       : {chunk.title}")
        print(f"Metadata    : {chunk.metadata}")
        print(f"Length      : {len(chunk.text)}")
        print(f"Text        : {chunk.text}")

    # 4. Basic validation
    assert text
    assert result.chunks
    assert all(
        chunk.text
        for chunk in result.chunks
    )

    assert all(
        chunk.source == str(TEST_FILE)
        for chunk in result.chunks
    )

    print()
    print("=" * 60)
    print("INTEGRATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()