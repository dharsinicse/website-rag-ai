from dataclasses import dataclass


@dataclass
class DocumentSection:
    heading: str
    text: str


class SectionExtractor:
    """Split a document into heading-based sections."""

    def extract(self, text: str) -> list[DocumentSection]:
        lines = text.splitlines()

        sections: list[DocumentSection] = []

        current_heading = ""
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            # Markdown-style headings.
            if stripped.startswith("#"):
                if current_lines:
                    sections.append(
                        DocumentSection(
                            heading=current_heading,
                            text="\n".join(current_lines).strip(),
                        )
                    )

                current_heading = stripped.lstrip("#").strip()
                current_lines = []

            else:
                current_lines.append(stripped)

        # Add the final section.
        if current_lines:
            sections.append(
                DocumentSection(
                    heading=current_heading,
                    text="\n".join(current_lines).strip(),
                )
            )

        return sections