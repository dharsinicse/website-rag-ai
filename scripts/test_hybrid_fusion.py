from retrieval.hybrid import ReciprocalRankFusion


vector_results = [
    {
        "source": "docs/a",
        "chunk_index": 0,
        "text": "Chunk A",
        "score": 0.90,
    },
    {
        "source": "docs/b",
        "chunk_index": 1,
        "text": "Chunk B",
        "score": 0.80,
    },
    {
        "source": "docs/c",
        "chunk_index": 2,
        "text": "Chunk C",
        "score": 0.70,
    },
]


keyword_results = [
    {
        "source": "docs/b",
        "chunk_index": 1,
        "text": "Chunk B",
        "score": 2.10,
    },
    {
        "source": "docs/a",
        "chunk_index": 0,
        "text": "Chunk A",
        "score": 1.50,
    },
    {
        "source": "docs/d",
        "chunk_index": 3,
        "text": "Chunk D",
        "score": 1.20,
    },
]


fusion = ReciprocalRankFusion(
    vector_weight=0.5,
    keyword_weight=0.5,
)

results = fusion.fuse(
    vector_results=vector_results,
    keyword_results=keyword_results,
    top_k=5,
)


print("HYBRID RESULTS")
print("=" * 70)

for rank, result in enumerate(results, start=1):
    print(
        f"{rank}. "
        f"fusion={result['fusion_score']:.6f} | "
        f"vector_rank={result['vector_rank']} | "
        f"keyword_rank={result['keyword_rank']} | "
        f"text={result['text']}"
    )