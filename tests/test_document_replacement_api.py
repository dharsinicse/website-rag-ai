
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


def test_replace_document_success_preserves_document_id(
    monkeypatch, tmp_path
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    old_path = upload_dir / "old.txt"
    old_path.write_text("Old document content", encoding="utf-8")

    metadata = [
        {
            "source": str(old_path),
            "metadata": {
                "document_id": "doc-123",
                "original_filename": "old.txt",
            },
        }
    ]

    settings = SimpleNamespace(
        upload_dir=str(upload_dir),
        max_upload_size_mb=5,
        vector_index_path=str(tmp_path / "vectors.faiss"),
        keyword_index_path=str(tmp_path / "keywords.pkl"),
    )

    captured = {}

    def fake_replace_document(**kwargs):
        captured.update(kwargs)
        return {
            "source": kwargs["new_file_path"],
            "chunks": 3,
            "status": "replaced",
        }

    monkeypatch.setattr(
        ingestion_routes, "get_settings", lambda: settings
    )
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(metadata),
    )
    monkeypatch.setattr(
        ingestion_routes,
        "get_ingestion_service",
        lambda: SimpleNamespace(
            replace_document=fake_replace_document
        ),
    )

    response = client.put(
        "/documents/doc-123/replace",
        files={
            "file": (
                "updated.txt",
                b"Updated document content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-123"
    assert body["filename"] == "updated.txt"
    assert body["status"] == "replaced"
    assert body["chunks"] == 3

    assert captured["old_source"] == str(old_path)
    assert captured["metadata"]["document_id"] == "doc-123"
    assert captured["metadata"]["original_filename"] == "updated.txt"
    assert old_path.exists() is False

    new_path = captured["new_file_path"]
    assert new_path != str(old_path)
    assert __import__("pathlib").Path(new_path).exists()


def test_replace_document_returns_404_for_unknown_id(monkeypatch):
    settings = SimpleNamespace(
        upload_dir="unused-uploads",
        max_upload_size_mb=5,
        vector_index_path="vectors.faiss",
        keyword_index_path="keywords.pkl",
    )

    monkeypatch.setattr(
        ingestion_routes, "get_settings", lambda: settings
    )
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline([]),
    )

    response = client.put(
        "/documents/missing-id/replace",
        files={
            "file": (
                "updated.txt",
                b"Updated content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_replace_document_preserves_old_file_on_failure(
    monkeypatch, tmp_path
):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    old_path = upload_dir / "old.txt"
    old_path.write_text("Old document content", encoding="utf-8")

    metadata = [
        {
            "source": str(old_path),
            "metadata": {
                "document_id": "doc-456",
                "original_filename": "old.txt",
            },
        }
    ]

    settings = SimpleNamespace(
        upload_dir=str(upload_dir),
        max_upload_size_mb=5,
        vector_index_path=str(tmp_path / "vectors.faiss"),
        keyword_index_path=str(tmp_path / "keywords.pkl"),
    )

    def failing_replace_document(**kwargs):
        raise RuntimeError("Simulated persistence failure")

    monkeypatch.setattr(
        ingestion_routes, "get_settings", lambda: settings
    )
    monkeypatch.setattr(
        ingestion_routes,
        "get_rag_pipeline",
        lambda: mock_pipeline(metadata),
    )
    monkeypatch.setattr(
        ingestion_routes,
        "get_ingestion_service",
        lambda: SimpleNamespace(
            replace_document=failing_replace_document
        ),
    )

    response = client.put(
        "/documents/doc-456/replace",
        files={
            "file": (
                "updated.txt",
                b"Updated content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Document replacement failed."
    assert old_path.exists()
    assert old_path.read_text(encoding="utf-8") == "Old document content"

    # The failed replacement upload must be cleaned up.
    assert list(upload_dir.iterdir()) == [old_path]
