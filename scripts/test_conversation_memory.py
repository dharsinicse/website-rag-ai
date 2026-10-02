from conversation.memory import ConversationMemory


def main():
    memory = ConversationMemory(max_messages=4)

    memory.add_message(
        "user",
        "What is the refund policy?",
    )

    memory.add_message(
        "assistant",
        "Refunds are available within 30 days.",
    )

    memory.add_message(
        "user",
        "How long does it take?",
    )

    memory.add_message(
        "assistant",
        "Refunds are generally processed within 5 business days.",
    )

    print("Conversation messages:")
    print("=" * 60)

    for message in memory.get_messages():
        print(f"{message['role']}: {message['content']}")

    print("\nMessage count:", len(memory))

    memory.clear()

    print("After clear:", len(memory))


if __name__ == "__main__":
    main()