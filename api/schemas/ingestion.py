from pydantic import BaseModel, Field

class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks: int = Field(ge=0)
    message: str