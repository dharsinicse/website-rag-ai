from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Website RAG AI"
    environment: str = "development"
    debug: bool = True

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    streamlit_host: str = "127.0.0.1"
    streamlit_port: int = 8501

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store: str = "faiss"

    
    vector_index_path: str = "storage/knowledge_base/vectors"
    keyword_index_path: str = "storage/knowledge_base/keywords.pkl"
    upload_dir: str = "data/raw/uploads"
    max_upload_size_mb: int = 10

    top_k: int = 5
    retrieval_candidates: int = 20

    chunk_size: int = 800
    chunk_overlap: int = 120

    max_crawl_pages: int = 50
    max_crawl_depth: int = 2

    llm_provider: str = "local"
    llm_model: str = ""

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()