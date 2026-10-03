from llm.providers.local import LocalLLMProvider


def main():
    model_name = "google/flan-t5-small"

    print("Loading model...")
    provider = LocalLLMProvider(
        model_name=model_name,
    )

    prompt = """
Answer the question using only the provided context.

Context:
Users can upload PDF and DOCX documents through the document upload page.

Question:
What documents can users upload?
"""

    print("\nGenerating answer...")

    answer = provider.generate(
        prompt=prompt,
        max_new_tokens=64,
    )

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()