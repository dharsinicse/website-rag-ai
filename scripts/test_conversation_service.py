from conversation.memory import ConversationMemory
from conversation.query_rewriter import SimpleQueryRewriter
from conversation.service import ConversationService


def main():
    memory = ConversationMemory(max_messages=10)

    rewriter = SimpleQueryRewriter()

    service = ConversationService(
        memory=memory,
        query_rewriter=rewriter,
    )

    # First user question
    first_query = "What is the refund policy?"

    rewritten_first = service.prepare_query(first_query)

    print("=" * 70)
    print("FIRST QUERY")
    print("Original :", first_query)
    print("Rewritten:", rewritten_first)

    service.add_user_message(first_query)

    # Simulated assistant response
    first_answer = "Refunds are available within 30 days."

    service.add_assistant_message(first_answer)

    # Follow-up question
    second_query = "How long does it take?"

    rewritten_second = service.prepare_query(second_query)

    print("\n" + "=" * 70)
    print("FOLLOW-UP QUERY")
    print("Original :", second_query)
    print("Rewritten:", rewritten_second)

    service.add_user_message(second_query)

    # Simulated assistant response
    second_answer = (
        "Refunds are generally processed within "
        "5 business days."
    )

    service.add_assistant_message(second_answer)

    print("\n" + "=" * 70)
    print("CONVERSATION HISTORY")

    for message in service.get_history():
        print(
            f"{message['role']}: "
            f"{message['content']}"
        )


if __name__ == "__main__":
    main()