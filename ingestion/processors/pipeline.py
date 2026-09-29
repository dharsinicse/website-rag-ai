from dataclasses import dataclass
from typing import Any

from ingestion.chunking.text_chunker import (
    TextChunk,
    TextChunker,
)
from ingestion.processors.cleaner import (
    DocumentCleaner,
)
from ingestion.processors.sections import (
    DocumentSection,
    SectionExtractor,
)


@dataclass
class ProcessedDocument:
    source: str
    title: str
    chunks: list[TextChunk]


class DocumentProcessor:
    """Process raw document text into RAG-ready chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        self.cleaner = DocumentCleaner()
        self.section_extractor = SectionExtractor()

        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process(
        self,
        text: str,
        source: str = "",
        title: str = "",
        page: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProcessedDocument:
        """Clean, section, and chunk a document."""

        # Step 1: Clean extracted text.
        cleaned_text = self.cleaner.clean(
            text
        )

        if not cleaned_text:
            return ProcessedDocument(
                source=source,
                title=title,
                chunks=[],
            )

        # Step 2: Extract sections.
        sections: list[DocumentSection] = (
            self.section_extractor.extract(
                cleaned_text
            )
        )

        # Step 3: Create chunks.
        chunks = self.chunker.chunk_sections(
            sections,
            source=source,
            title=title,
            page=page,
            metadata=metadata,
        )

        return ProcessedDocument(
            source=source,
            title=title,
            chunks=chunks,
        )