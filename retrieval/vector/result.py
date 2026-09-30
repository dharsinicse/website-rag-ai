from dataclasses import dataclass
from typing import Any


@dataclass
class VectorSearchResult:
    """A single vector search result."""

    score: float
    text: str
    source: str
    title: str = ""
    heading: str = ""
    page: int | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] | None = None