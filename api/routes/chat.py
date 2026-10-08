from fastapi import APIRouter, HTTPException

from api.rag_app import get_rag_pipeline
from api.schemas.chat import ChatRequest, ChatResponse, Source


router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """Answer a user query using the RAG pipeline."""

    try:
        pipeline = get_rag_pipeline()

        result = pipeline.answer(
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
        )

        sources = [
            Source(
                source=source.get("source", ""),
                title=source.get("title", ""),
                heading=source.get("heading", ""),
                page=source.get("page"),
            )
            for source in result.get("sources", [])
        ]

        return ChatResponse(
            query=result["query"],
            answer=result["answer"],
            sources=sources,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline failed while processing the request.",
        ) from exc