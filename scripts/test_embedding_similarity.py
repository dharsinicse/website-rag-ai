from sklearn.metrics.pairwise import cosine_similarity

from embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)


def main():
    print("Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding()

    texts = [
        "How can I log into the application?",
        "Users authenticate using their account credentials.",
        "The weather is sunny today.",
    ]

    embeddings = embedding_model.embed_texts(
        texts
    )

    similarity_matrix = cosine_similarity(
        embeddings
    )

    print()
    print("=" * 60)
    print("EMBEDDING SIMILARITY TEST")
    print("=" * 60)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = similarity_matrix[i][j]

            print()
            print(f"Text {i}: {texts[i]}")
            print(f"Text {j}: {texts[j]}")
            print(
                f"Cosine similarity: "
                f"{similarity:.4f}"
            )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()