"""Data Transfer Objects for API contracts.

These Pydantic models define the external API contract.
They are used for request/response validation and serialization.

Internal domain logic should use entities from the entities package.
"""

from .requests import CacheDeleteRequest, CheckCacheRequest, StoreCacheRequest
from .responses import (
    CacheCheckResponse,
    CacheMatchItem,
    CacheStatsResponse,
    CacheStoreResponse,
    HealthCheckResponse,
)

__all__ = [
    "CheckCacheRequest",
    "StoreCacheRequest",
    "CacheDeleteRequest",
    "CacheMatchItem",
    "CacheCheckResponse",
    "CacheStoreResponse",
    "CacheStatsResponse",
    "HealthCheckResponse",
]
