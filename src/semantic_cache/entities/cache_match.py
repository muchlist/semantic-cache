"""Cache match domain entity."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheMatchEntity:
    """Domain entity for a cache search result.

    Represents a single match from a vector similarity search.

    Attributes:
        prompt: The matched prompt from the cache
        response: The cached response
        distance: Cosine distance (0 = identical, 2 = opposite)
        cached_at: Timestamp when the entry was created (Unix timestamp)
        metadata: Optional metadata stored with the cache entry
    """

    prompt: str
    response: str
    distance: float
    cached_at: float
    metadata: dict[str, Any] | None = None
