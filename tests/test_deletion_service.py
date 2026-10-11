
import pytest

from ingestion.service import IngestionService
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


class FakeEmbeddingService:
    dimension = 3


def build_service():
    vector_store = FAISSVectorStore(dimension=3)
    vector_store.add(
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        metadata=[
            {"text": "alpha document", "source": "alpha.txt"},
            {"text": "beta document", "source": "beta.txt"},
        ],
    )

    keyword_store = BM25KeywordStore()
    keyword_store.add(
        texts=["alpha document", "beta document"],
        metadata=[
            {"text": "alpha document", "source": "alpha.txt"},
            {"text": "beta document", "source": "beta.txt"},
        ],
    )

    return IngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
        keyword_store=keyword_store,
    )


def test_delete_document_updates_both_stores_and_persists(
    tmp_path,
):
    service = build_service()

    result = service.delete_document(
        source="alpha.txt",
        vector_path=str(tmp_path / "vectors"),
        keyword_path=str(tmp_path / "keywords.pkl"),
    )

    assert result["status"] == "deleted"
    assert result["deleted_chunks"] == 1
    assert [item["source"] for item in service.vector_store.metadata] == [
        "beta.txt"
    ]
    assert [item["source"] for item in service.keyword_store.metadata] == [
        "beta.txt"
    ]
    assert (tmp_path / "vectors").exists()
    assert (tmp_path / "vectors.metadata.npy").exists()
    assert (tmp_path / "keywords.pkl").exists()


def test_delete_document_restores_memory_if_persistence_fails(
    tmp_path,
    monkeypatch,
):
    import ingestion.service as service_module

    service = build_service()
    original_vector_index = service.vector_store.index
    original_vector_metadata = list(service.vector_store.metadata)
    original_documents = list(service.keyword_store.documents)

    def fail_save(**kwargs):
        raise OSError("Simulated persistence failure")

    monkeypatch.setattr(
        service_module,
        "save_staged_indexes",
        fail_save,
    )

    with pytest.raises(OSError, match="Simulated persistence failure"):
        service.delete_document(
            source="alpha.txt",
            vector_path=str(tmp_path / "vectors"),
            keyword_path=str(tmp_path / "keywords.pkl"),
        )

    assert service.vector_store.index is original_vector_index
    assert service.vector_store.metadata == original_vector_metadata
    assert service.keyword_store.documents == original_documents