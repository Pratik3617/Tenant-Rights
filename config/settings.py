from pydantic_settings import BaseSettings

class settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_EMBED_MODEL : str = "text-embedding-3-large"
    OPENAI_EMBED_DIMS : int = 3072
    OPENAI_LLM_MODEL : str = "gpt-4o"
    OPENAI_FAST_MODEL : str = "gpt-4o-mini"

    # weaviate
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_INDEX: str = "TenantRightsChunk"
    WEAVIATE_API_KEY: str | None = None

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600  # Cache Time-To-Live in seconds

    # Postgres
    DATABASE_URL: str

    # Langsmith
    LANGSMITH_API_KEY: str
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "tenant-rights_rag"

    class Config:
        env_file = ".env"

settings = settings()