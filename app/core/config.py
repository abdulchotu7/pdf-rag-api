from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    embedding_dimension: int = 1024

    chunk_size: int = 512
    chunk_overlap: int = 64

    top_k: int = 10

    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 1024
    retrieval_top_k: int = 20
    final_top_k: int = 5

    hybrid_enabled: bool = True
    vector_top_k: int = 10
    bm25_top_k: int = 10
    fusion_top_k: int = 15

    upload_dir: str = "uploads"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_name: str = "PDF RAG API"
    app_version: str = "0.1.0"
    app_timeout: int = 600
    max_upload_size: int = 1073741824

    log_level: str = "INFO"


settings = Settings()
