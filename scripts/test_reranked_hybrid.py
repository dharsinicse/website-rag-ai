from embeddings.sentence_transformer import SentenceTransformerEmbedding
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.hybrid import (
    HybridRetriever,
    RerankedHybridRetriever,
    ReciprocalRankFusion,
)
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector.faiss_store import FAISSVectorStore


def main():
    chunks = [
        {
            "text": "Users can upload PDF and DOCX documents through the document upload page.",
            "source": "https://example.com/docs",
            "title": "Product Documentation",
            "heading": "Documents",
            "page": None,
            "chunk_index": 0,
            "metadata": {},
        },
        {
            "text": "Users can reset their password from account settings.",
            "source": "https://example.com/docs",
            "title": "Product Documentation",
            "heading": "Password Reset",
            "page": None,
            "chunk_index": 1,
            "metadata": {},
        },
        {
            "text": "Authentication requires a valid email address.",
            "source": "https://example.com/docs",
            "title": "Product Documentation",
            "heading": "Authentication",
            "page": None,
            "chunk_index": 2,
            "metadata": {},
        },
        {
            "text": "The API returns HTTP 401 when authentication fails.",
            "source": "https://example.com/api",
            "title": "API Documentation",
            "heading": "API Errors",
            "page": None,
            "chunk_index": 3,
            "metadata": {},
        },
        {
            "text": "The dashboard displays weather information for the selected city.",
            "source": "https://example.com/docs",
            "title": "Product Documentation",
            "heading": "Dashboard",
            "page": None,
            "chunk_index": 4,
            "metadata": {},
        },
    ]

    embedding_model = SentenceTransformerEmbedding()

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedding_model.embed_texts(texts)

    vector_store = FAISSVectorStore(
        dimension=embedding_model.dimension
    )

    vector_store.add(
        vectors=vectors,
        metadata=chunks,
    )

    keyword_store = BM25KeywordStore()

    keyword_store.add(
        texts=texts,
        metadata=chunks,
    )

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_store=keyword_store,
        fusion=ReciprocalRankFusion(),
    )

    reranker = CrossEncoderReranker()

    retriever = RerankedHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    query = "How can users upload a PDF?"

    query_vector = embedding_model.embed_text(query)

    results = retriever.search(
        query=query,
        query_vector=query_vector,
        top_k=3,
        candidate_k=5,
    )

    print("\nQuery:")
    print(query)

    print("\nFinal Reranked Results:")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"rerank_score={result['rerank_score']:.4f} | "
            f"heading={result['heading']}"
        )
        print(f"   {result['text']}")


if __name__ == "__main__":
    main()