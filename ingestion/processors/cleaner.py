import re

class DocumentCleaner:
    """Clean and normalize extracted document text."""

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Normalize line endings.
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive spaces and tabs.
        text = re.sub(r"[ \t]+", " ", text)

        # Remove spaces around line breaks.
        text = re.sub(r" *\n *", "\n", text)

        # Collapse more than two consecutive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()