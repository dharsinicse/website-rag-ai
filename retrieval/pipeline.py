from embeddings.service import EmbeddingService
from retrieval.hybrid import RerankedHybridRetriever
from llm.generator import RAGGenerator


class RAGPipeline:
    """End-to-end retrieval augmented generation pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        retriever: RerankedHybridRetriever,
        generator: RAGGenerator,
    ):
        self.embedding_service = embedding_service
        self.retriever = retriever
        self.generator = generator

    def answer(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        max_new_tokens: int = 256,
    ) -> dict:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Step 1: Convert the query into an embedding.
        query_vector = self.embedding_service.embed_text(query)

        # Step 2: Retrieve candidate documents.
        results = self.retriever.search(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        # Step 3: Generate a grounded answer.
        generation_result = self.generator.generate(
            query=query,
            results=results,
            max_new_tokens=max_new_tokens,
        )

        return {
            "query": query,
            "answer": generation_result["answer"],
            "sources": generation_result["sources"],
            "retrieved_documents": results,
        }