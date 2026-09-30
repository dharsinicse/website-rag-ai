from embeddings.sentence_transformer import SentenceTransformerEmbedding
from ingestion.chunking.text_chunker import TextChunk
from retrieval.hybrid import HybridRetriever, ReciprocalRankFusion
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.vector.faiss_store import FAISSVectorStore


chunks = [
    TextChunk(
        text="Users can upload PDF and DOCX documents.",
        chunk_index=0,
        heading="Documents",
        source="https://example.com/docs",
        title="Documentation",
    ),
    TextChunk(
        text="Users can reset their password from account settings.",
        chunk_index=1,
        heading="Password Reset",
        source="https://example.com/security",
        title="Security",
    ),
    TextChunk(
        text="Authentication requires a valid email address.",
        chunk_index=2,
        heading="Authentication",
        source="https://example.com/auth",
        title="Authentication",
    ),
    TextChunk(
        text="The API returns HTTP 401 when authentication fails.",
        chunk_index=3,
        heading="API Errors",
        source="https://example.com/api",
        title="API Reference",
    ),
    TextChunk(
        text="The dashboard displays weather information for the selected city.",
        chunk_index=4,
        heading="Dashboard",
        source="https://example.com/dashboard",
        title="Dashboard",
    ),
]


print("Loading embedding model...")

embedding_model = SentenceTransformerEmbedding()

print("Embedding documents...")

texts = [chunk.text for chunk in chunks]

embeddings = embedding_model.embed_texts(texts)

print(f"Created {len(embeddings)} embeddings")
print(f"Embedding dimension: {embedding_model.dimension}")


# --------------------------------------------------
# FAISS
# --------------------------------------------------

vector_store = FAISSVectorStore(
    dimension=embedding_model.dimension
)

vector_metadata = [
    {
        "text": chunk.text,
        "source": chunk.source,
        "title": chunk.title,
        "heading": chunk.heading,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "metadata": chunk.metadata,
    }
    for chunk in chunks
]

vector_store.add(
    vectors=embeddings,
    metadata=vector_metadata,
)


# --------------------------------------------------
# BM25
# --------------------------------------------------

keyword_store = BM25KeywordStore()

keyword_metadata = [
    {
        "source": chunk.source,
        "title": chunk.title,
        "heading": chunk.heading,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "metadata": chunk.metadata,
    }
    for chunk in chunks
]

keyword_store.add(
    texts=texts,
    metadata=keyword_metadata,
)


# --------------------------------------------------
# Hybrid Retriever
# --------------------------------------------------

fusion = ReciprocalRankFusion(
    vector_weight=0.5,
    keyword_weight=0.5,
)

retriever = HybridRetriever(
    vector_store=vector_store,
    keyword_store=keyword_store,
    fusion=fusion,
)


# --------------------------------------------------
# Query
# --------------------------------------------------

queries = [
    "How can users upload a PDF?",
    "HTTP 401 authentication error",
    "How do I reset my password?",
    "What email is required for authentication?",
    "weather dashboard",
]

for query in queries:

    print()
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    query_vector = embedding_model.embed_text(query)

    results = retriever.search(
        query_vector=query_vector,
        query=query,
        top_k=3,
        candidate_k=5,
    )

    print()
    print("TOP HYBRID RESULTS")
    print("-" * 70)

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"fusion={result['fusion_score']:.6f} | "
            f"vector_rank={result['vector_rank']} | "
            f"keyword_rank={result['keyword_rank']}"
        )

        print(f"   heading={result['heading']}")
        print(f"   text={result['text']}")
        print()