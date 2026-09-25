from pathlib import Path

import pymupdf
from docx import Document


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = []

    with pymupdf.open(path) as pdf:
        for page in pdf:
            page_text = page.get_text()

            if page_text:
                text.append(page_text)

    return "\n".join(text).strip()


def load_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    document = Document(path)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def load_text(file_path: str) -> str:
    """Load TXT or Markdown files."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()


def load_html(file_path: str) -> str:
    """Extract readable text from an HTML file."""

    from bs4 import BeautifulSoup

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    html = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    soup = BeautifulSoup(html, "lxml")

    for element in soup(
        ["script", "style", "noscript"]
    ):
        element.decompose()

    return soup.get_text(
        separator="\n",
        strip=True,
    )