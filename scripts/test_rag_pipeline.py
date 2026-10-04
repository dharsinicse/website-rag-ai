from embeddings.service import EmbeddingService
from embeddings.sentence_transformer import SentenceTransformerEmbedding

from retrieval.vector.faiss_store import FAISSVectorStore
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.hybrid import HybridRetriever, RerankedHybridRetriever
from retrieval.reranker import CrossEncoderReranker

from llm.providers.local import LocalLLMProvider
from llm.prompts import RAGPromptBuilder
from llm.generator import RAGGenerator

from retrieval.pipeline import RAGPipeline


def main():
    print("=" * 70)
    print("END-TO-END RAG PIPELINE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load embedding model
    # ---------------------------------------------------------
    print("\n[1/7] Loading embedding model...")

    embedding_model = SentenceTransformerEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_service = EmbeddingService(
        embedding_model
    )

    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    # ---------------------------------------------------------
    # 2. Prepare sample knowledge
    # ---------------------------------------------------------
    print("\n[2/7] Preparing knowledge base...")

    documents = [
        {
            "text": (
                "Users can upload PDF and DOCX documents "
                "through the document upload interface."
            ),
            "source": "https://example.com/docs/uploads",
            "title": "Document Upload Guide",
            "heading": "Supported Documents",
            "chunk_index": 0,
        },
        {
            "text": (
                "The platform supports authentication "
                "using email and password."
            ),
            "source": "https://example.com/docs/auth",
            "title": "Authentication Guide",
            "heading": "Login",
            "chunk_index": 0,
        },
        {
            "text": (
                "Users can reset their password from "
                "the account settings page."
            ),
            "source": "https://example.com/docs/password",
            "title": "Password Guide",
            "heading": "Password Reset",
            "chunk_index": 0,
        },
    ]

    texts = [
        document["text"]
        for document in documents
    ]

    vectors = embedding_service.embed_texts(texts)

    print(f"Documents: {len(documents)}")
    print(f"Vectors: {len(vectors)}")

    # ---------------------------------------------------------
    # 3. Build FAISS vector store
    # ---------------------------------------------------------
    print("\n[3/7] Building vector store...")

    vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    vector_store.add(
        vectors=vectors,
        metadata=documents,
    )

    print("FAISS vector store ready.")

    # ---------------------------------------------------------
    # 4. Build BM25 keyword store
    # ---------------------------------------------------------
    print("\n[4/7] Building keyword store...")

    keyword_store = BM25KeywordStore()

    keyword_store.add(
        texts=texts,
        metadata=documents,
    )

    print("BM25 keyword store ready.")

    # ---------------------------------------------------------
    # 5. Build hybrid + reranker
    # ---------------------------------------------------------
    print("\n[5/7] Building retrieval pipeline...")

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_store=keyword_store,
    )

    reranker = CrossEncoderReranker(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    retriever = RerankedHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    print("Hybrid retrieval + reranking ready.")

    # ---------------------------------------------------------
    # 6. Build LLM generator
    # ---------------------------------------------------------
    print("\n[6/7] Loading LLM...")

    llm = LocalLLMProvider(
        model_name="google/flan-t5-small"
    )

    prompt_builder = RAGPromptBuilder()

    generator = RAGGenerator(
        llm=llm,
        prompt_builder=prompt_builder,
    )

    print("LLM generator ready.")

    # ---------------------------------------------------------
    # 7. Build complete RAG pipeline
    # ---------------------------------------------------------
    print("\n[7/7] Building complete RAG pipeline...")

    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        retriever=retriever,
        generator=generator,
    )

    query = "How can users upload a PDF?"

    print("\n" + "=" * 70)
    print("USER QUERY")
    print("=" * 70)

    print(query)

    result = pipeline.answer(
        query=query,
        top_k=3,
        candidate_k=3,
        max_new_tokens=128,
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result["sources"]:
        print(
            f"[{source['id']}] "
            f"{source['title']} - "
            f"{source['source']}"
        )

    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    for index, document in enumerate(
        result["retrieved_documents"],
        start=1,
    ):
        print(
            f"{index}. "
            f"{document.get('title', 'Unknown')} "
            f"| Rerank score: "
            f"{document.get('rerank_score', 0):.4f}"
        )

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()