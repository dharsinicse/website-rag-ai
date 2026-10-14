
from unittest.mock import Mock

import pytest

from ingestion.service import IngestionService
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


class FakeEmbeddingService:
    def embed_chunks(self, chunks):
        return [
            [1.0, 0.0, 0.0]
            for _ in chunks
        ]


def build_service():
    vector_store = FAISSVectorStore(dimension=3)
    vector_store.add(
        vectors=[[0.0, 1.0, 0.0]],
        metadata=[
            {
                "text": "Old product information",
                "source": "old.txt",
                "metadata": {"document_id": "old"},
            }
        ],
    )

    keyword_store = BM25KeywordStore()
    keyword_store.add(
        texts=["Old product information"],
        metadata=[
            {
                "text": "Old product information",
                "source": "old.txt",
                "metadata": {"document_id": "old"},
            }
        ],
    )

    return IngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        keyword_store=keyword_store,
    )


def test_replace_document_updates_both_indexes(
    tmp_path, monkeypatch
):
    service = build_service()
    new_file = tmp_path / "new.txt"
    new_file.write_text(
        "Updated product information",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        service,
        "load_file",
        lambda _: "Updated product information",
    )
    monkeypatch.setattr(
        "ingestion.service.save_staged_indexes",
        lambda **kwargs: None,
    )

    result = service.replace_document(
        old_source="old.txt",
        new_file_path=str(new_file),
        metadata={"document_id": "new"},
        vector_path=str(tmp_path / "vectors"),
        keyword_path=str(tmp_path / "keywords.pkl"),
    )

    assert result["status"] == "replaced"
    assert result["chunks"] > 0
    assert all(
        item["source"] != "old.txt"
        for item in service.vector_store.metadata
    )
    assert all(
        item["source"] != "old.txt"
        for item in service.keyword_store.metadata
    )
    assert any(
        item["source"] == str(new_file)
        for item in service.vector_store.metadata
    )
    assert any(
        item["source"] == str(new_file)
        for item in service.keyword_store.metadata
    )


def test_replace_document_restores_stores_if_persistence_fails(
    tmp_path, monkeypatch
):
    service = build_service()
    new_file = tmp_path / "new.txt"
    new_file.write_text(
        "Updated product information",
        encoding="utf-8",
    )

    original_index = service.vector_store.index
    original_vector_metadata = list(service.vector_store.metadata)
    original_documents = list(service.keyword_store.documents)
    original_keyword_metadata = list(service.keyword_store.metadata)

    monkeypatch.setattr(
        service,
        "load_file",
        lambda _: "Updated product information",
    )

    def fail_save(**kwargs):
        raise OSError("Simulated persistence failure")

    monkeypatch.setattr(
        "ingestion.service.save_staged_indexes",
        fail_save,
    )

    with pytest.raises(
        OSError,
        match="Simulated persistence failure",
    ):
        service.replace_document(
            old_source="old.txt",
            new_file_path=str(new_file),
            metadata={"document_id": "new"},
            vector_path=str(tmp_path / "vectors"),
            keyword_path=str(tmp_path / "keywords.pkl"),
        )

    assert service.vector_store.index is original_index
    assert service.vector_store.metadata == original_vector_metadata
    assert service.keyword_store.documents == original_documents
    assert service.keyword_store.metadata == original_keyword_metadata
