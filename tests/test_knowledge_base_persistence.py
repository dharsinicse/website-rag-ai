import pytest
import retrieval.knowledge_base_persistence as persistence

from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore
from retrieval.knowledge_base_persistence import save_staged_indexes


def test_save_staged_indexes_can_be_loaded_again(tmp_path):
    vector_path = tmp_path / "vectors"
    keyword_path = tmp_path / "keywords.pkl"

    vector_store = FAISSVectorStore(dimension=3)
    vector_store.add(
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
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

    save_staged_indexes(
        vector_store=vector_store,
        keyword_store=keyword_store,
        vector_path=str(vector_path),
        keyword_path=str(keyword_path),
    )

    assert vector_path.exists()
    assert vector_path.with_suffix(".metadata.npy").exists()
    assert keyword_path.exists()

    loaded_vector_store = FAISSVectorStore(dimension=3)
    loaded_vector_store.load(str(vector_path))

    loaded_keyword_store = BM25KeywordStore()
    loaded_keyword_store.load(str(keyword_path))

    assert loaded_vector_store.index.ntotal == 2
    assert len(loaded_vector_store.metadata) == 2
    assert len(loaded_keyword_store.documents) == 2
    assert [item["source"] for item in loaded_vector_store.metadata] == [
        "alpha.txt",
        "beta.txt",
    ]
    assert [item["source"] for item in loaded_keyword_store.metadata] == [
        "alpha.txt",
        "beta.txt",
    ]


def test_failed_replacement_restores_existing_files(
    tmp_path,
    monkeypatch,
):
    vector_path = tmp_path / "vectors"
    keyword_path = tmp_path / "keywords.pkl"
    metadata_path = vector_path.with_suffix(".metadata.npy")

    # Simulate an existing persisted knowledge base.
    vector_path.write_bytes(b"old-vector-index")
    metadata_path.write_bytes(b"old-vector-metadata")
    keyword_path.write_bytes(b"old-keyword-index")

    vector_store = FAISSVectorStore(dimension=3)
    vector_store.add(
        vectors=[[1.0, 0.0, 0.0]],
        metadata=[{"text": "new document", "source": "new.txt"}],
    )

    keyword_store = BM25KeywordStore()
    keyword_store.add(
        texts=["new document"],
        metadata=[{"text": "new document", "source": "new.txt"}],
    )

    real_replace = persistence.os.replace
    call_count = 0

    def fail_on_second_replace(source, destination):
        nonlocal call_count
        call_count += 1

        if call_count == 2:
            raise OSError("Simulated replacement failure")

        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_on_second_replace)

    with pytest.raises(OSError, match="Simulated replacement failure"):
        persistence.save_staged_indexes(
            vector_store=vector_store,
            keyword_store=keyword_store,
            vector_path=str(vector_path),
            keyword_path=str(keyword_path),
        )

    assert vector_path.read_bytes() == b"old-vector-index"
    assert metadata_path.read_bytes() == b"old-vector-metadata"
    assert keyword_path.read_bytes() == b"old-keyword-index"
