
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
import api.routes.ingestion as ingestion_routes


client = TestClient(app)


def mock_pipeline(metadata):
    keyword_store = SimpleNamespace(metadata=metadata)
    hybrid_retriever = SimpleNamespace(keyword_store=keyword_store)
    retriever = SimpleNamespace(hybrid_retriever=hybrid_retriever)
    return SimpleNamespace(retriever=retriever)


def test_list_documents_empty(monkeypatch):
    pipeline = mock_pipeline([])
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: pipeline,
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": [], "total": 0}


def test_list_documents_groups_chunks(monkeypatch):
    metadata = [
        {
            "source": "data/uploads/doc1.txt",
            "metadata": {
                "document_id": "doc-1",
                "original_filename": "guide.txt",
            },
        },
        {
            "source": "data/uploads/doc1.txt",
            "metadata": {
                "document_id": "doc-1",
                "original_filename": "guide.txt",
            },
        },
        {
            "source": "data/uploads/doc2.pdf",
            "metadata": {
                "document_id": "doc-2",
                "original_filename": "manual.pdf",
            },
        },
    ]

    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(metadata),
    )

    response = client.get("/documents")

    assert response.status_code == 200

    result = response.json()
    assert result["total"] == 2

    documents = {
        document["document_id"]: document
        for document in result["documents"]
    }

    assert documents["doc-1"]["filename"] == "guide.txt"
    assert documents["doc-1"]["source"] == "data/uploads/doc1.txt"
    assert documents["doc-1"]["chunks"] == 2

    assert documents["doc-2"]["filename"] == "manual.pdf"
    assert documents["doc-2"]["chunks"] == 1


def test_list_documents_uses_source_name_as_filename_fallback(monkeypatch):
    metadata = [
        {
            "source": "data/uploads/legacy.txt",
            "metadata": {"document_id": "doc-legacy"},
        }
    ]

    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(metadata),
    )

    response = client.get("/documents")

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["filename"] == "legacy.txt"
    assert document["chunks"] == 1
