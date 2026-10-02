from conversation.query_rewriter import SimpleQueryRewriter


def main():
    rewriter = SimpleQueryRewriter()

    history = [
        {
            "role": "user",
            "content": "What is the refund policy?",
        },
        {
            "role": "assistant",
            "content": "Refunds are available within 30 days.",
        },
    ]

    queries = [
        "What is the refund policy?",
        "How long does it take?",
        "Can I get a refund?",
    ]

    for query in queries:
        rewritten = rewriter.rewrite(
            query=query,
            conversation_history=history,
        )

        print("=" * 60)
        print(f"Original : {query}")
        print(f"Rewritten: {rewritten}")


if __name__ == "__main__":
    main()