from retrieval.keyword.bm25_store import BM25KeywordStore


documents = [
    "Users can upload PDF and DOCX documents.",
    "Users can reset their password from the account settings.",
    "Authentication requires a valid email address.",
    "Weather information is available on the dashboard.",
]

metadata = [
    {
        "source": "docs/upload",
        "title": "Document Upload",
        "heading": "Documents",
    },
    {
        "source": "docs/security",
        "title": "Security",
        "heading": "Password Reset",
    },
    {
        "source": "docs/auth",
        "title": "Authentication",
        "heading": "Login",
    },
    {
        "source": "docs/weather",
        "title": "Weather",
        "heading": "Weather",
    },
]


store = BM25KeywordStore()

store.add(
    texts=documents,
    metadata=metadata,
)

results = store.search(
    query="PDF upload",
    top_k=3,
)

print(f"Found {len(results)} results")
print("=" * 60)

for rank, result in enumerate(results, start=1):
    print(f"Rank: {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Heading: {result['heading']}")
    print(f"Text: {result['text']}")
    print("-" * 60)