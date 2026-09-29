from embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)


def main():
    print("Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding()

    texts = [
        "Our platform supports semantic search.",
        "Users can upload PDF and DOCX documents.",
        "The system provides grounded AI answers.",
        "Website pages can be crawled automatically.",
    ]

    embeddings = embedding_model.embed_texts(
        texts
    )

    print()
    print("=" * 60)
    print("BATCH EMBEDDING TEST")
    print("=" * 60)

    print(f"Model: {embedding_model.model_name}")
    print(f"Texts: {len(texts)}")
    print(f"Embeddings: {len(embeddings)}")
    print(f"Dimension: {embedding_model.dimension}")

    print()

    for index, embedding in enumerate(
        embeddings
    ):
        print(
            f"Text {index}: "
            f"vector length = {len(embedding)}"
        )

    print()
    print("=" * 60)

    assert len(embeddings) == len(texts)

    assert all(
        len(embedding)
        == embedding_model.dimension
        for embedding in embeddings
    )

    print("BATCH EMBEDDING TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()