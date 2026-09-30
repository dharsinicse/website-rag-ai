from pathlib import Path
from typing import Any

import faiss
import numpy as np

from retrieval.vector.base import VectorStore
from retrieval.vector.result import VectorSearchResult


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError(
                "Dimension must be greater than 0"
            )

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.metadata: list[dict[str, Any]] = []

    def add(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add vectors and their metadata."""

        if not vectors:
            return

        if len(vectors) != len(metadata):
            raise ValueError(
                "Number of vectors must match "
                "number of metadata entries"
            )

        array = np.asarray(
            vectors,
            dtype="float32",
        )

        if array.ndim != 2:
            raise ValueError(
                "Vectors must be a 2D array"
            )

        if array.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vector dimension "
                f"{self.dimension}, "
                f"got {array.shape[1]}"
            )

        faiss.normalize_L2(array)

        self.index.add(array)

        self.metadata.extend(metadata)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Search for the most similar vectors."""

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if self.index.ntotal == 0:
            return []

        query = np.asarray(
            [query_vector],
            dtype="float32",
        )

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension "
                f"{self.dimension}, "
                f"got {query.shape[1]}"
            )

        faiss.normalize_L2(query)

        k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query,
            k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            metadata = self.metadata[index]

            result = VectorSearchResult(
                score=float(score),
                text=metadata.get(
                    "text",
                    "",
                ),
                source=metadata.get(
                    "source",
                    "",
                ),
                title=metadata.get(
                    "title",
                    "",
                ),
                heading=metadata.get(
                    "heading",
                    "",
                ),
                page=metadata.get(
                    "page",
                    None,
                ),
                chunk_index=metadata.get(
                    "chunk_index",
                    0,
                ),
                metadata=metadata.get(
                    "metadata",
                    {},
                ),
            )

            results.append(result)

        return results

    def save(self, path: str) -> None:
        """Save FAISS index and metadata."""

        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(output_path),
        )

        metadata_path = output_path.with_suffix(
            ".metadata.npy"
        )

        np.save(
            metadata_path,
            np.array(
                self.metadata,
                dtype=object,
            ),
            allow_pickle=True,
        )

    def load(self, path: str) -> None:
        """Load FAISS index and metadata."""

        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {path}"
            )

        self.index = faiss.read_index(
            str(input_path)
        )

        metadata_path = input_path.with_suffix(
            ".metadata.npy"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS metadata file not found: "
                f"{metadata_path}"
            )

        self.metadata = np.load(
            metadata_path,
            allow_pickle=True,
        ).tolist()