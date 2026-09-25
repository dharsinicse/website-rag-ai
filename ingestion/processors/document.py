from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    text: str

    source: str

    title: str = ""

    page: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)