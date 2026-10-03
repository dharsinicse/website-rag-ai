from llm.providers.local import LocalLLMProvider
from llm.prompts import RAGPromptBuilder


def main():
    model_name = "google/flan-t5-small"

    print("Loading LLM...")
    llm = LocalLLMProvider(
        model_name=model_name,
    )

    prompt_builder = RAGPromptBuilder()

    context = """
Document: User Guide

Users can upload PDF and DOCX documents
through the document upload page.

PDF files are processed automatically.
DOCX files are converted into text before indexing.
"""

    query = "What documents can users upload?"

    prompt = prompt_builder.build(
        query=query,
        context=context,
    )

    print("\nGenerating RAG answer...")

    answer = llm.generate(
        prompt=prompt,
        max_new_tokens=64,
    )

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    main()