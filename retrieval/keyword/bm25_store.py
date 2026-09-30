import pickle
import re
from typing import Any

from rank_bm25 import BM25Okapi

from retrieval.keyword.base import KeywordStore


class BM25KeywordStore(KeywordStore):
    """BM25-based keyword retrieval store."""

    def __init__(self):
        self.documents: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.tokenized_documents: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Convert text into normalized tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def add(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add documents to the BM25 index."""

        if len(texts) != len(metadata):
            raise ValueError(
                "texts and metadata must have the same length"
            )

        if not texts:
            return

        self.documents.extend(texts)
        self.metadata.extend(metadata)

        self.tokenized_documents = [
            self._tokenize(text)
            for text in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search indexed documents using BM25."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            results.append(
                {
                    "score": float(scores[index]),
                    "text": self.documents[index],
                    **self.metadata[index],
                }
            )

        return results

    def save(self, path: str) -> None:
        """Save BM25 index to disk."""

        with open(path, "wb") as file:
            pickle.dump(
                {
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "tokenized_documents": self.tokenized_documents,
                },
                file,
            )

    def load(self, path: str) -> None:
        """Load BM25 index from disk."""

        with open(path, "rb") as file:
            data = pickle.load(file)

        self.documents = data["documents"]
        self.metadata = data["metadata"]
        self.tokenized_documents = data["tokenized_documents"]

        if self.tokenized_documents:
            self.bm25 = BM25Okapi(
                self.tokenized_documents
            )
        else:
            self.bm25 = None