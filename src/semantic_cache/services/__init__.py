"""Service layer for business logic.

This layer contains the core business logic and orchestration.
Services depend on protocols (interfaces), not concrete implementations,
making them testable and flexible.

Architecture:
    Handler -> Service -> Repository
    (HTTP)  -> (Business) -> (Data Access)

Usage:
    ```python
    from semantic_cache.services import CacheService

    # Using factory method (recommended)
    cache = CacheService.create()
    cache = CacheService.create(distance_threshold=0.2)

    # Or manual creation
    cache = CacheService(repository=repo, embedding_provider=provider)
    ```
"""

from .cache_service import CacheService

__all__ = [
    "CacheService",
]
