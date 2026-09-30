from ingestion.chunking.text_chunker import TextChunk
from retrieval.keyword.bm25_store import BM25KeywordStore
from retrieval.keyword.service import KeywordRetrievalService


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


store = BM25KeywordStore()
service = KeywordRetrievalService(store)

service.index_chunks(chunks)


queries = [
    "PDF upload",
    "password",
    "HTTP 401",
    "authentication",
    "weather",
]


for query in queries:

    print()
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = service.search(
        query=query,
        top_k=3,
    )

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"score={result['score']:.4f} | "
            f"{result['heading']} | "
            f"{result['text']}"
        )