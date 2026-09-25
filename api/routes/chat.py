from fastapi import APIRouter


router = APIRouter()


@router.post("/chat")
def chat():
    return {
        "message": "Chat endpoint is ready."
    }