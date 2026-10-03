from llm.base import LLMProvider
from llm.prompts import RAGPromptBuilder


class RAGGenerator:
    """Generate grounded answers from retrieved documents."""

    def __init__(
        self,
        llm: LLMProvider,
        prompt_builder: RAGPromptBuilder,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder

    def generate(
        self,
        query: str,
        results: list[dict],
        max_new_tokens: int = 256,
    ) -> dict:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not results:
            return {
                "answer": (
                    "I don't have enough information "
                    "in the provided sources."
                ),
                "sources": [],
            }

        context_parts = []
        sources = []

        for index, result in enumerate(results, start=1):
            text = result.get("text", "").strip()

            if not text:
                continue

            source = result.get("source", "Unknown source")
            title = result.get("title", "")
            heading = result.get("heading", "")

            context_parts.append(
                f"[Source {index}]\n"
                f"Title: {title}\n"
                f"Source: {source}\n"
                f"Section: {heading}\n"
                f"Content: {text}"
            )

            sources.append(
                {
                    "id": index,
                    "source": source,
                    "title": title,
                    "heading": heading,
                }
            )

        if not context_parts:
            return {
                "answer": (
                    "I don't have enough information "
                    "in the provided sources."
                ),
                "sources": [],
            }

        context = "\n\n".join(context_parts)

        prompt = self.prompt_builder.build(
            query=query,
            context=context,
        )

        answer = self.llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        return {
            "answer": answer,
            "sources": sources,
        }