"""Protocol interfaces for swappable implementations.

This package contains protocol definitions using structural typing.
Protocols enable:
- Easy swapping of implementations (Redis → PostgreSQL, local → OpenAI, etc.)
- Unit testing with mock implementations
- Clear separation of concerns

Usage:
    ```python
    from semantic_cache.repositories.protocols import CacheStore, EmbeddingProvider

    # Type hints work with any implementation
    repo: CacheStore = RedisCacheRepository()  # works
    repo: CacheStore = PgVectorRepository()     # also works
    ```
"""

from .cache_store import CacheStore
from .embedding_provider import EmbeddingProvider

__all__ = [
    "CacheStore",
    "EmbeddingProvider",
]
