from pydantic import BaseModel, Field

class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks: int = Field(ge=0)
    message: str

class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    source: str
    chunks: int = Field(ge=0)

class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int = Field(ge=0)
    limit: int = Field(gt=0, le=100)
    offset: int = Field(ge=0)