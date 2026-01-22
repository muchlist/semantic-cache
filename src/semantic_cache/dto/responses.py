"""Response DTOs for API endpoints."""

from pydantic import BaseModel, Field


class CacheMatchItem(BaseModel):
    """Single cache match item (in matches array)."""

    prompt: str = Field(..., description="The matched prompt from cache")
    response: str = Field(..., description="The cached response")
    vector_distance: float = Field(
        ...,
        description="Cosine distance (0 = identical, 2 = opposite)",
        ge=0.0,
        le=2.0,
    )
    cosine_similarity: float = Field(
        ...,
        description="Cosine similarity (1 = identical, 0 = opposite)",
        ge=0.0,
        le=1.0,
    )
    cached_at: float = Field(..., description="Timestamp when the entry was cached (Unix timestamp)")


class CacheCheckResponse(BaseModel):
    """Response DTO for cache check operation.

    Matches the legacy API format with:
    - Original prompt
    - is_hit boolean
    - matches array (can have multiple results)
    - lookup_time_ms
    """

    prompt: str = Field(..., description="The original query prompt")
    is_hit: bool = Field(..., description="Whether any cache entries matched within threshold")
    matches: list[CacheMatchItem] = Field(
        default_factory=list,
        description="List of matched cache entries (sorted by distance, closest first)",
    )
    lookup_time_ms: float = Field(..., description="Time taken for the cache lookup in milliseconds")


class CacheStoreResponse(BaseModel):
    """Response DTO for cache store operation."""

    success: bool = Field(..., description="Whether the operation succeeded")
    key: str = Field(..., description="The storage key for the entry")
    message: str = Field(..., description="Human-readable status message")


class CacheStatsResponse(BaseModel):
    """Response DTO for cache statistics."""

    total_entries: int = Field(
        ...,
        description="Total number of cached entries",
        ge=0,
    )
    index_name: str = Field(..., description="Name of the vector index")
    threshold: float = Field(
        ...,
        description="Current similarity threshold",
        ge=0.0,
        le=2.0,
    )
    ttl_seconds: int = Field(
        ...,
        description="Time-to-live for cache entries in seconds",
        ge=0,
    )


class HealthCheckResponse(BaseModel):
    """Response DTO for health check."""

    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    cache_healthy: bool = Field(..., description="Whether the cache backend is reachable")
    embedding_healthy: bool | None = Field(
        None,
        description="Whether the embedding service is reachable",
    )
