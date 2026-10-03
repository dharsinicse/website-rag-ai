from llm.generator import RAGGenerator
from llm.prompts import RAGPromptBuilder
from llm.providers.local import LocalLLMProvider


def main():
    print("Loading LLM...")

    llm = LocalLLMProvider(
        model_name="google/flan-t5-small",
    )

    generator = RAGGenerator(
        llm=llm,
        prompt_builder=RAGPromptBuilder(),
    )

    result = generator.generate(
        query="What is the company's vacation policy?",
        results=[],
    )

    print("\n" + "=" * 70)
    print("EMPTY RETRIEVAL TEST")
    print("=" * 70)
    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    print(result["sources"])


if __name__ == "__main__":
    main()