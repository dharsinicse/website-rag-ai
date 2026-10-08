from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question for the RAG system",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of final documents to retrieve",
    )
    candidate_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of retrieval candidates",
    )


class Source(BaseModel):
    source: str
    title: str = ""
    heading: str = ""
    page: int | None = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]