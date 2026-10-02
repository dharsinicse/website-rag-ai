from retrieval.reranker import CrossEncoderReranker


def main():
    query = "How can users upload a PDF?"

    results = [
        {
            "text": "The dashboard displays weather information for the selected city.",
            "source": "https://example.com/docs",
            "heading": "Dashboard",
        },
        {
            "text": "Users can upload PDF and DOCX documents through the document upload page.",
            "source": "https://example.com/docs",
            "heading": "Documents",
        },
        {
            "text": "Users can reset their password from account settings.",
            "source": "https://example.com/docs",
            "heading": "Password Reset",
        },
        {
            "text": "The API returns HTTP 401 when authentication fails.",
            "source": "https://example.com/api",
            "heading": "API Errors",
        },
    ]

    reranker = CrossEncoderReranker()

    reranked = reranker.rerank(
        query=query,
        results=results,
        top_k=3,
    )

    print("\nQuery:")
    print(query)

    print("\nReranked Results:")
    print("=" * 60)

    for rank, result in enumerate(reranked, start=1):
        print(
            f"{rank}. "
            f"score={result['rerank_score']:.4f} | "
            f"heading={result['heading']}"
        )
        print(f"   {result['text']}")


if __name__ == "__main__":
    main()