from retrieval.hybrid import RerankedHybridRetriever


class FakeHybridRetriever:
    """Return a known-good hybrid ranking."""

    def search(
        self,
        query_vector,
        query,
        top_k=5,
        candidate_k=20,
    ):
        return [
            {
                "text": "Correct Search document",
                "source": "test.txt",
                "title": "Test",
                "heading": "Search",
                "chunk_index": 0,
                "vector_score": 0.9,
                "keyword_score": 2.0,
            },
            {
                "text": "Security document",
                "source": "test.txt",
                "title": "Test",
                "heading": "Security",
                "chunk_index": 1,
                "vector_score": 0.4,
                "keyword_score": 1.0,
            },
        ]


class FakeBadReranker:
    """Return an intentionally incorrect ranking."""

    def rerank(
        self,
        query,
        results,
        top_k=5,
    ):
        return [
            {
                **results[1],
                "rerank_score": 5.0,
            },
            {
                **results[0],
                "rerank_score": -5.0,
            },
        ][:top_k]


def test_reranker_safety_falls_back_to_hybrid():
    hybrid = FakeHybridRetriever()
    bad_reranker = FakeBadReranker()

    retriever = RerankedHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=bad_reranker,
    )

    results = retriever.search(
        query="How does the system search documents?",
        query_vector=[0.1, 0.2, 0.3],
        top_k=2,
        candidate_k=10,
    )

    assert results[0]["text"] == "Correct Search document"

    assert results[0]["heading"] == "Search"

    assert results[1]["text"] == "Security document"