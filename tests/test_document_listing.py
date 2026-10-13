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

def sample_metadata():
    return [
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
        {
            "source": "data/uploads/doc3.txt",
            "metadata": {
                "document_id": "doc-3",
                "original_filename": "tutorial.txt",
            },
        },
    ]

def test_list_documents_empty(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline([]),
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {
        "documents": [],
        "total": 0,
        "limit": 20,
        "offset": 0,
    }

def test_list_documents_groups_chunks(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(sample_metadata()),
    )

    response = client.get("/documents")

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 3

    documents = {
        document["document_id"]: document
        for document in result["documents"]
    }

    assert documents["doc-1"]["filename"] == "guide.txt"
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


def test_list_documents_pagination(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(sample_metadata()),
    )

    response = client.get("/documents?limit=1&offset=1")

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 3
    assert result["limit"] == 1
    assert result["offset"] == 1
    assert len(result["documents"]) == 1
    assert result["documents"][0]["document_id"] == "doc-2"


def test_list_documents_search_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(sample_metadata()),
    )

    response = client.get("/documents?search=GUIDE")

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 1
    assert result["documents"][0]["filename"] == "guide.txt"


def test_list_documents_filters_by_file_type(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(sample_metadata()),
    )

    response = client.get("/documents?file_type=.txt")

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 2
    assert {doc["filename"] for doc in result["documents"]} == {
        "guide.txt",
        "tutorial.txt",
    }


def test_list_documents_combines_filters(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(sample_metadata()),
    )

    response = client.get("/documents?search=guide&file_type=txt")

    assert response.status_code == 200
    result = response.json()
    assert result["total"] == 1
    assert result["documents"][0]["document_id"] == "doc-1"


def test_list_documents_rejects_invalid_pagination(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(sample_metadata()),
    )

    for query in ("limit=0", "limit=101", "offset=-1"):
        response = client.get(f"/documents?{query}")
        assert response.status_code == 422, query


def test_list_documents_rejects_blank_search(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(sample_metadata()),
    )

    response = client.get("/documents?search=%20%20%20")

    assert response.status_code == 422

def test_get_document_returns_details_and_chunk_count(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(sample_metadata()),
    )

    response = client.get("/documents/doc-1")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-1",
        "filename": "guide.txt",
        "source": "data/uploads/doc1.txt",
        "chunks": 2,
    }


def test_get_document_uses_source_name_as_filename_fallback(monkeypatch):
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

    response = client.get("/documents/doc-legacy")

    assert response.status_code == 200
    assert response.json()["filename"] == "legacy.txt"
    assert response.json()["chunks"] == 1


def test_get_document_returns_404_for_unknown_id(monkeypatch):
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(sample_metadata()),
    )

    response = client.get("/documents/does-not-exist")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."