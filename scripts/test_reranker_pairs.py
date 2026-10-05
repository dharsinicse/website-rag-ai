from sentence_transformers import CrossEncoder


def main():
    print("=" * 60)
    print("CROSS-ENCODER PAIR DIAGNOSTIC")
    print("=" * 60)

    model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"

    print("\nLoading reranker...")
    model = CrossEncoder(model_name)

    query = "How does the system search documents?"

    passages = [
        (
            "CORRECT",
            "The platform uses hybrid retrieval combining "
            "semantic vector search and keyword search."
        ),
        (
            "SECURITY",
            "Uploaded documents are validated before processing. "
            "The system also applies limits to protect against "
            "oversized files and unsafe content."
        ),
        (
            "UPLOAD",
            "Users can upload PDF, DOCX, TXT, and Markdown "
            "documents to the platform."
        ),
        (
            "AUTHENTICATION",
            "Users must authenticate before accessing "
            "protected documents and account features."
        ),
    ]

    pairs = [
        (query, passage)
        for _, passage in passages
    ]

    scores = model.predict(pairs)

    print("\nQUERY")
    print("-" * 60)
    print(query)

    print("\nDIRECT PAIR SCORES")
    print("-" * 60)

    for (label, passage), score in zip(
        passages,
        scores,
    ):
        print(f"\n{label}")
        print(f"Score: {float(score):.6f}")
        print(f"Text: {passage}")

    print("\n" + "=" * 60)
    print("PAIR DIAGNOSTIC COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()