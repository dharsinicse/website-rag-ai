
from pydantic import BaseModel, Field


class IngestionResponse(BaseModel):
    filename: str
    status: str
    chunks: int = Field(ge=0)
    message: str