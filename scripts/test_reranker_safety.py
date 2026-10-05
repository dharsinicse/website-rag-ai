from retrieval.hybrid import RerankedHybridRetriever


class FakeHybridRetriever:
    """Simulate a correct hybrid retrieval ranking."""

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
    """Simulate a reranker that incorrectly changes the top result."""

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


def main():
    print("=" * 60)
    print("DAY 12 - RERANKER SAFETY REGRESSION TEST")
    print("=" * 60)

    print("\n[1/4] Creating hybrid retriever...")
    hybrid = FakeHybridRetriever()

    print("[2/4] Creating intentionally bad reranker...")
    bad_reranker = FakeBadReranker()

    print("[3/4] Building safe reranked retriever...")

    retriever = RerankedHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=bad_reranker,
    )

    print("[4/4] Running safety test...")

    results = retriever.search(
        query="How does the system search documents?",
        query_vector=[0.1, 0.2, 0.3],
        top_k=2,
        candidate_k=10,
    )

    print("\nHybrid expected top result:")
    print(results[0]["text"])

    if results[0]["text"] != "Correct Search document":
        raise AssertionError(
            "Safety guard failed: incorrect reranked result "
            "was returned."
        )

    if results[0]["heading"] != "Search":
        raise AssertionError(
            "Safety guard failed: Search result was not "
            "restored."
        )

    print("\nPASS: Safety guard rejected the bad reranking.")

    print("\nFinal ranking:")

    for index, result in enumerate(results, start=1):
        print(
            f"{index}. "
            f"{result['text']} "
            f"(heading={result['heading']})"
        )

    print("\n" + "=" * 60)
    print("RERANKER SAFETY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()