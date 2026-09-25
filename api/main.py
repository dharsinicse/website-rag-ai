from fastapi import FastAPI

from config.settings import get_settings
from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.ingestion import router as ingestion_router


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Industry-ready RAG API",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingestion_router)