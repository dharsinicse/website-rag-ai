from llm.generator import RAGGenerator
from llm.prompts import RAGPromptBuilder
from llm.providers.local import LocalLLMProvider


def main():
    print("Loading LLM...")

    llm = LocalLLMProvider(
        model_name="google/flan-t5-small",
    )

    prompt_builder = RAGPromptBuilder()

    generator = RAGGenerator(
        llm=llm,
        prompt_builder=prompt_builder,
    )

    results = [
        {
            "text": (
                "Users can upload PDF and DOCX documents "
                "through the document upload page."
            ),
            "source": "https://example.com/user-guide",
            "title": "User Guide",
            "heading": "Document Upload",
        },
        {
            "text": (
                "PDF files are processed automatically "
                "before being indexed."
            ),
            "source": "https://example.com/user-guide",
            "title": "User Guide",
            "heading": "PDF Processing",
        },
    ]

    query = "What documents can users upload?"

    print("\nGenerating answer...")

    result = generator.generate(
        query=query,
        results=results,
        max_new_tokens=64,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result["sources"]:
        print(f"[{source['id']}] {source['title']}")
        print(f"    URL: {source['source']}")
        print(f"    Section: {source['heading']}")


if __name__ == "__main__":
    main()