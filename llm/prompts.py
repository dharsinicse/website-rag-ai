class RAGPromptBuilder:
    """Build grounded prompts for RAG answer generation."""

    SYSTEM_INSTRUCTION = """
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context.

Rules:
1. Do not use information outside the context.
2. Do not invent or assume facts.
3. If the answer cannot be found in the context, say:
   "I don't have enough information in the provided sources."
4. Give a concise and direct answer.
5. When possible, mention the relevant source.
""".strip()

    def build(
        self,
        query: str,
        context: str,
    ) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not context.strip():
            raise ValueError("Context cannot be empty")

        return f"""
{self.SYSTEM_INSTRUCTION}

Context:
{context}

User Question:
{query}

Answer:
""".strip()