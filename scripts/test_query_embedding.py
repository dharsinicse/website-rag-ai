from embeddings.service import EmbeddingService
from embeddings.sentence_transformer import SentenceTransformerEmbedding


def main():
    print("Loading embedding model...")

    model = SentenceTransformerEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    service = EmbeddingService(model)

    query = "How can users upload a PDF?"

    embedding = service.embed_text(query)

    print("\n" + "=" * 70)
    print("QUERY EMBEDDING TEST")
    print("=" * 70)

    print(f"Query: {query}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()