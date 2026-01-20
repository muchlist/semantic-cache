import os
from dataclasses import dataclass
from functools import lru_cache

import redis
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_password: str | None = os.getenv("REDIS_PASSWORD")

    # Cache
    cache_distance_threshold: float = float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.15"))
    cache_ttl: int = int(os.getenv("CACHE_TTL", "604800"))  # 7 days default
    cache_index_name: str = os.getenv("CACHE_INDEX_NAME", "semantic_cache")

    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_reload: bool = os.getenv("API_RELOAD", "true").lower() == "true"

    # Optional: OpenAI
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    def __post_init__(self) -> None:
        """Validate settings after initialization."""
        if not 0 <= self.cache_distance_threshold <= 2:
            raise ValueError("CACHE_DISTANCE_THRESHOLD must be between 0 and 2 for cosine distance")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


def get_redis_client() -> redis.Redis:
    """Create a Redis client instance."""
    return redis.from_url(
        settings.redis_url,
        password=settings.redis_password,
        decode_responses=False,
    )
