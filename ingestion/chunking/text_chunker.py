from dataclasses import dataclass, field
from typing import Any
import re


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    heading: str = ""

    source: str = ""
    title: str = ""
    page: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class TextChunker:
    """Create overlapping chunks while preserving metadata."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        """Split text into approximate sentences."""

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _split_long_text(
        self,
        text: str,
    ) -> list[str]:
        """Split long text on word boundaries."""

        words = text.split()

        pieces = []
        current_words = []
        current_length = 0

        for word in words:
            additional_length = (
                len(word)
                if not current_words
                else len(word) + 1
            )

            if (
                current_words
                and current_length + additional_length
                > self.chunk_size
            ):
                pieces.append(
                    " ".join(current_words)
                )

                current_words = []
                current_length = 0

            current_words.append(word)

            current_length += (
                len(word)
                if current_length == 0
                else len(word) + 1
            )

        if current_words:
            pieces.append(
                " ".join(current_words)
            )

        return pieces

    def chunk_section(
        self,
        text: str,
        heading: str = "",
        start_index: int = 0,
        source: str = "",
        title: str = "",
        page: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Create chunks while preserving metadata."""

        text = text.strip()

        if not text:
            return []

        sentences = self._split_sentences(text)

        expanded_sentences = []

        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                expanded_sentences.extend(
                    self._split_long_text(sentence)
                )
            else:
                expanded_sentences.append(sentence)

        chunks = []

        current_sentences = []
        current_length = 0
        chunk_index = start_index

        for sentence in expanded_sentences:
            sentence_length = len(sentence)

            additional_length = (
                sentence_length
                if not current_sentences
                else sentence_length + 1
            )

            if (
                current_sentences
                and current_length + additional_length
                > self.chunk_size
            ):
                chunk_text = " ".join(
                    current_sentences
                ).strip()

                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        heading=heading,
                        source=source,
                        title=title,
                        page=page,
                        metadata=metadata.copy()
                        if metadata
                        else {},
                    )
                )

                chunk_index += 1

                overlap_sentences = []
                overlap_length = 0

                for previous in reversed(
                    current_sentences
                ):
                    previous_length = (
                        len(previous)
                        if not overlap_sentences
                        else len(previous) + 1
                    )

                    if (
                        overlap_length
                        + previous_length
                        > self.chunk_overlap
                    ):
                        break

                    overlap_sentences.insert(
                        0,
                        previous,
                    )

                    overlap_length += previous_length

                current_sentences = overlap_sentences

                current_length = overlap_length

            current_sentences.append(sentence)

            current_length = len(
                " ".join(current_sentences)
            )

        if current_sentences:
            chunks.append(
                TextChunk(
                    text=" ".join(
                        current_sentences
                    ).strip(),
                    chunk_index=chunk_index,
                    heading=heading,
                    source=source,
                    title=title,
                    page=page,
                    metadata=metadata.copy()
                    if metadata
                    else {},
                )
            )

        return chunks

    def chunk_sections(
        self,
        sections,
        source: str = "",
        title: str = "",
        page: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Create chunks from document sections."""

        all_chunks = []
        chunk_index = 0

        for section in sections:
            chunks = self.chunk_section(
                text=section.text,
                heading=section.heading,
                start_index=chunk_index,
                source=source,
                title=title,
                page=page,
                metadata=metadata,
            )

            all_chunks.extend(chunks)

            chunk_index += len(chunks)

        return all_chunks