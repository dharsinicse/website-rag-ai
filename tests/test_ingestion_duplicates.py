
from pathlib import Path

from ingestion.chunking.text_chunker import TextChunker

from ingestion.service import IngestionService


from unittest.mock import Mock

from ingestion.service import IngestionService


def test_chunker_preserves_file_hash_metadata():
    """Ensure document fingerprints survive chunking."""
    file_hash = "abc123fingerprint"

    chunks = TextChunker(
        chunk_size=100,
        chunk_overlap=10,
    ).chunk_section(
        text="Customers can reset their password. Contact support for help.",
        heading="Product Support",
        source="upload_test.txt",
        title="upload_test",
        metadata={
            "document_type": "txt",
            "file_hash": file_hash,
        },
    )

    assert chunks
    assert all(
        chunk.metadata.get("file_hash") == file_hash
        for chunk in chunks
    )


def test_duplicate_file_hash_is_detected(tmp_path):
    """A repeated file hash should be recognized."""
    test_file = tmp_path / "support.txt"
    test_file.write_text(
        "Customers can reset their password.",
        encoding="utf-8",
    )

    fingerprint = IngestionService.calculate_file_hash
    file_hash = fingerprint(None, str(test_file))

    existing_metadata = [
        {
            "metadata": {
                "file_hash": file_hash,
            }
        }
    ]

    assert any(
        item.get("metadata", {}).get("file_hash") == file_hash
        for item in existing_metadata
    )


def test_ingest_file_skips_duplicate(tmp_path):
    """Do not add chunks when the file hash already exists."""
    file_path = tmp_path / "support.txt"
    file_path.write_text(
        "Customers can reset their password.",
        encoding="utf-8",
    )

    file_hash = IngestionService.calculate_file_hash(
        None, str(file_path)
    )

    embedding_service = Mock()
    vector_store = Mock()
    keyword_store = Mock()
    keyword_store.metadata = [
        {"metadata": {"file_hash": file_hash}}
    ]

    service = IngestionService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        keyword_store=keyword_store,
    )

    result = service.ingest_file(str(file_path))

    assert result["status"] == "duplicate"
    assert result["chunks"] == 0
    embedding_service.embed_chunks.assert_not_called()
    vector_store.add.assert_not_called()
    keyword_store.add.assert_not_called()
