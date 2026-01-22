"""Request DTOs for API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class CheckCacheRequest(BaseModel):
    """Request DTO for checking cache.

    The handler will convert this to internal calls to the service layer.
    """

    prompt: str = Field(..., description="The prompt to search for", min_length=1)
    threshold: float | None = Field(
        None,
        description="Override the default distance threshold (0-2, lower = more strict)",
        ge=0.0,
        le=2.0,
    )


class StoreCacheRequest(BaseModel):
    """Request DTO for storing in cache."""

    prompt: str = Field(..., description="The original user prompt", min_length=1)
    response: str = Field(..., description="The LLM response to cache", min_length=1)
    metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Optional metadata (model name, tokens, cost, etc.)",
    )


class CacheDeleteRequest(BaseModel):
    """Request DTO for deleting cache entries."""

    prompt: str | None = Field(
        None,
        description="Delete entries by exact prompt match (if null, clears all)",
    )


# Alias for convenience when clearing all
ClearCacheRequest = CacheDeleteRequest
