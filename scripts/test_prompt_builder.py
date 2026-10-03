from llm.prompts import RAGPromptBuilder


def main():
    builder = RAGPromptBuilder()

    context = """
Document: User Guide

Users can upload PDF and DOCX documents
through the document upload page.

PDF files are processed automatically.
DOCX files are converted into text before indexing.
"""

    query = "What documents can users upload?"

    prompt = builder.build(
        query=query,
        context=context,
    )

    print("=" * 70)
    print("GENERATED RAG PROMPT")
    print("=" * 70)
    print(prompt)


if __name__ == "__main__":
    main()