"""Repository layer for data access.

This layer abstracts external dependencies (Redis, databases, embedding APIs)
behind protocol-based interfaces. This enables:
- Easy swapping of implementations (Redis → PostgreSQL, local → OpenAI, etc.)
- Unit testing with mock implementations
- Clear separation of concerns

The repositories are protocol-based (structural typing), not inheritance-based.
Any class implementing the required methods will satisfy the protocol.
"""

from semantic_cache.protocols import CacheStore, EmbeddingProvider

from .local_embedding_provider import LocalEmbeddingProvider
from .redis_repository import RedisCacheRepository

# Re-export with backward compatible names
CacheRepository = CacheStore

__all__ = [
    "CacheStore",
    "CacheRepository",  # Backward compatible alias
    "EmbeddingProvider",
    "RedisCacheRepository",
    "LocalEmbeddingProvider",
]
