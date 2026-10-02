from typing import Any
from retrieval.reranker import CrossEncoderReranker

class ReciprocalRankFusion:
    """Combine ranked retrieval results using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        k: int = 60,
    ):
        if vector_weight < 0:
            raise ValueError("vector_weight cannot be negative")

        if keyword_weight < 0:
            raise ValueError("keyword_weight cannot be negative")

        if vector_weight == 0 and keyword_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than 0"
            )

        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.k = k

    def _result_key(self, result: dict[str, Any]) -> str:
        """Create a stable identifier for a retrieved chunk."""

        source = result.get("source", "")
        chunk_index = result.get("chunk_index", 0)

        return f"{source}::{chunk_index}"

    def fuse(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Fuse vector and keyword rankings."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        fused: dict[str, dict[str, Any]] = {}

        # Add semantic/vector results.
        for rank, result in enumerate(vector_results, start=1):
            key = self._result_key(result)

            if key not in fused:
                fused[key] = {
                    **result,
                    "vector_rank": rank,
                    "keyword_rank": None,
                    "vector_score": result.get("score", 0.0),
                    "keyword_score": 0.0,
                    "fusion_score": 0.0,
                }

            fused[key]["vector_rank"] = rank
            fused[key]["vector_score"] = result.get(
                "score",
                0.0,
            )

            fused[key]["fusion_score"] += (
                self.vector_weight
                / (self.k + rank)
            )

        # Add keyword/BM25 results.
        for rank, result in enumerate(keyword_results, start=1):
            key = self._result_key(result)

            if key not in fused:
                fused[key] = {
                    **result,
                    "vector_rank": None,
                    "keyword_rank": rank,
                    "vector_score": 0.0,
                    "keyword_score": result.get("score", 0.0),
                    "fusion_score": 0.0,
                }

            fused[key]["keyword_rank"] = rank
            fused[key]["keyword_score"] = result.get(
                "score",
                0.0,
            )

            fused[key]["fusion_score"] += (
                self.keyword_weight
                / (self.k + rank)
            )

        ranked_results = sorted(
            fused.values(),
            key=lambda result: result["fusion_score"],
            reverse=True,
        )

        return ranked_results[:top_k]

from retrieval.vector.base import VectorStore
from retrieval.keyword.base import KeywordStore


class HybridRetriever:
    """Retrieve documents using both semantic and keyword search."""

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        fusion: ReciprocalRankFusion | None = None,
    ):
        self.vector_store = vector_store
        self.keyword_store = keyword_store

        self.fusion = fusion or ReciprocalRankFusion()

    def search(
        self,
        query_vector: list[float],
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Run vector + keyword retrieval and fuse the results."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not query_vector:
            raise ValueError("Query vector cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_k,
        )

        vector_results = [
            {
                "score": result.score,
                "text": result.text,
                "source": result.source,
                "title": result.title,
                "heading": result.heading,
                "page": result.page,
                "chunk_index": result.chunk_index,
                "metadata": result.metadata,
            }
            for result in vector_results
        ]
        
        keyword_results = self.keyword_store.search(
            query=query,
            top_k=candidate_k,
        )

        return self.fusion.fuse(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=top_k,
        )

class RerankedHybridRetriever:
    """Combine hybrid retrieval with Cross-Encoder reranking."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        candidates = self.hybrid_retriever.search(
            query_vector=query_vector,
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=top_k,
        )