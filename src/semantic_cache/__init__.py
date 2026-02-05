"""Semantic Cache - Multilingual semantic caching with vector search.

This package provides a layered architecture for semantic caching:

Layers:
    - protocols: Interface contracts (CacheStore, EmbeddingProvider)
    - repositories: Data access implementations
    - services: Business logic
    - handlers: HTTP endpoint handlers
    - dto: Data transfer objects (API contracts)
    - entities: Domain models (internal)

Usage:
    ```python
    from semantic_cache.services import CacheService

    # Using class method (recommended, like Path.home())
    cache = CacheService.create()
    cache = CacheService.create(distance_threshold=0.2)
    ```

For HTTP API:
    ```python
    from semantic_cache.api.app import app
    ```
"""

from semantic_cache.config import get_redis_client, settings
from semantic_cache.dto import CheckCacheRequest, StoreCacheRequest
from semantic_cache.entities import CacheEntryEntity, CacheMatchEntity
from semantic_cache.handlers import CacheHandler

# New architecture exports
from semantic_cache.protocols import CacheStore, EmbeddingProvider
from semantic_cache.repositories import RedisCacheRepository
from semantic_cache.services import CacheService

__all__ = [
    # Configuration
    "settings",
    "get_redis_client",
    # Protocols (interfaces)
    "CacheStore",
    "EmbeddingProvider",
    # Services (business logic)
    "CacheService",
    # Handlers (HTTP)
    "CacheHandler",
    # Repositories (data access)
    "RedisCacheRepository",
    "LocalEmbeddingProvider",
    # Entities (domain models)
    "CacheEntryEntity",
    "CacheMatchEntity",
    # DTOs (API contracts)
    "CheckCacheRequest",
    "StoreCacheRequest",
]
