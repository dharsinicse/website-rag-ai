
import numpy as np
import pytest

from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


def make_metadata(source, text):
    return {
        "source": source,
        "title": source,
        "text": text,
        "chunk_index": 0,
        "metadata": {},
    }


def test_faiss_delete_by_source():
    store = FAISSVectorStore(dimension=3)

    store.add(
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        metadata=[
            make_metadata("a.txt", "alpha"),
            make_metadata("b.txt", "bravo"),
            make_metadata("a.txt", "alpha second chunk"),
        ],
    )

    deleted = store.delete_by_source("a.txt")

    assert deleted == 2
    assert store.index.ntotal == 1
    assert len(store.metadata) == 1
    assert store.metadata[0]["source"] == "b.txt"


def test_faiss_delete_missing_source():
    store = FAISSVectorStore(dimension=3)
    store.add(
        vectors=[[1.0, 0.0, 0.0]],
        metadata=[make_metadata("a.txt", "alpha")],
    )

    deleted = store.delete_by_source("missing.txt")

    assert deleted == 0
    assert store.index.ntotal == 1
    assert len(store.metadata) == 1


def test_bm25_delete_by_source():
    store = BM25KeywordStore()
    store.add(
        texts=[
            "alpha customer support",
            "bravo billing information",
            "alpha password reset",
        ],
        metadata=[
            make_metadata("a.txt", "alpha customer support"),
            make_metadata("b.txt", "bravo billing information"),
            make_metadata("a.txt", "alpha password reset"),
        ],
    )

    deleted = store.delete_by_source("a.txt")

    assert deleted == 2
    assert store.documents == ["bravo billing information"]
    assert len(store.metadata) == 1
    assert store.metadata[0]["source"] == "b.txt"
    assert len(store.tokenized_documents) == 1
    assert store.bm25 is not None


def test_bm25_delete_last_document():
    store = BM25KeywordStore()
    store.add(
        texts=["alpha customer support"],
        metadata=[make_metadata("a.txt", "alpha customer support")],
    )

    deleted = store.delete_by_source("a.txt")

    assert deleted == 1
    assert store.documents == []
    assert store.metadata == []
    assert store.tokenized_documents == []
    assert store.bm25 is None


@pytest.mark.parametrize("source", ["", None])
def test_delete_rejects_empty_source(source):
    vector_store = FAISSVectorStore(dimension=3)
    keyword_store = BM25KeywordStore()

    with pytest.raises(ValueError):
        vector_store.delete_by_source(source)

    with pytest.raises(ValueError):
        keyword_store.delete_by_source(source)
