import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from semantic_cache.cache import SemanticCache
from semantic_cache.config import get_redis_client, settings
from semantic_cache.embeddings import EmbeddingService
from semantic_cache.models import CacheResult, PerformanceMetrics


class CacheCheckRequest(BaseModel):
    """Request model for cache check endpoint."""

    prompt: str
    distance_threshold: float | None = None


class CacheCheckResponse(BaseModel):
    """Response model for cache check endpoint."""

    prompt: str
    is_hit: bool
    matches: list[dict[str, Any]]
    lookup_time_ms: float


class CacheStoreRequest(BaseModel):
    """Request model for cache store endpoint."""

    prompt: str
    response: str
    metadata: dict[str, Any] | None = None


class CacheStoreResponse(BaseModel):
    """Response model for cache store endpoint."""

    success: bool
    key: str
    message: str


class HydrateRequest(BaseModel):
    """Request model for cache hydration endpoint."""

    pairs: list[tuple[str, str]]
    metadata: dict[str, Any] | None = None


class HydrateResponse(BaseModel):
    """Response model for cache hydration endpoint."""

    success: bool
    count: int
    message: str


class ThresholdRequest(BaseModel):
    """Request model for setting threshold endpoint."""

    threshold: float


# Global instances
_cache_instance: SemanticCache | None = None
_perf_metrics = PerformanceMetrics()


def get_cache() -> SemanticCache:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    print("Starting Semantic Cache API...")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"Distance threshold: {settings.cache_distance_threshold}")

    # Test Redis connection
    try:
        client = get_redis_client()
        client.ping()
        print("Redis connection successful")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        print("Make sure Redis is running: docker compose up -d")

    yield

    # Shutdown
    print("Shutting down Semantic Cache API...")


app = FastAPI(
    title="Semantic Cache API",
    description="Semantic caching service using Redis and sentence-transformers",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": "Semantic Cache API",
        "version": "0.1.0",
        "description": "Semantic caching service using Redis and sentence-transformers",
        "endpoints": {
            "cache": "/cache",
            "stats": "/stats",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    try:
        client = get_redis_client()
        client.ping()
        return {
            "status": "healthy",
            "redis": "connected",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis connection failed: {e}",
        )


@app.post("/cache/check", response_model=CacheCheckResponse)
async def check_cache(request: CacheCheckRequest) -> CacheCheckResponse:
    """
    Check cache for semantically similar prompts.

    Args:
        request: Cache check request with prompt and optional threshold.

    Returns:
        Cache check response with matches and lookup time.
    """
    cache = get_cache()
    start_time = time.time()

    try:
        result: CacheResult = cache.check(request.prompt, request.distance_threshold)
        lookup_time_ms = (time.time() - start_time) * 1000

        # Update performance metrics
        if result.is_hit:
            _perf_metrics.record_hit(lookup_time_ms)
        else:
            _perf_metrics.record_miss(lookup_time_ms)

        return CacheCheckResponse(
            prompt=result.prompt,
            is_hit=result.is_hit,
            matches=[
                {
                    "prompt": m.prompt,
                    "response": m.response,
                    "vector_distance": m.vector_distance,
                    "cosine_similarity": m.cosine_similarity,
                    "cached_at": m.cached_at,
                }
                for m in result.matches
            ],
            lookup_time_ms=lookup_time_ms,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache check failed: {e}",
        )


@app.post("/cache/store", response_model=CacheStoreResponse)
async def store_cache(request: CacheStoreRequest) -> CacheStoreResponse:
    """
    Store a prompt/response pair in the cache.

    Args:
        request: Cache store request with prompt, response, and optional metadata.

    Returns:
        Cache store response with success status and key.
    """
    cache = get_cache()

    try:
        key = cache.store(request.prompt, request.response, request.metadata)
        return CacheStoreResponse(
            success=True,
            key=key,
            message=f"Stored cache entry: {key}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache store failed: {e}",
        )


@app.post("/cache/hydrate", response_model=HydrateResponse)
async def hydrate_cache(request: HydrateRequest) -> HydrateResponse:
    """
    Hydrate cache from a list of prompt/response pairs.

    Args:
        request: Hydration request with pairs and optional metadata.

    Returns:
        Hydration response with success status and count.
    """
    cache = get_cache()

    try:
        count = cache.hydrate_from_pairs(request.pairs, request.metadata)
        return HydrateResponse(
            success=True,
            count=count,
            message=f"Hydrated {count} cache entries",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache hydration failed: {e}",
        )


@app.delete("/cache", response_model=dict[str, str])
async def clear_cache() -> dict[str, str]:
    """Clear all entries from the cache."""
    cache = get_cache()

    try:
        cache.clear()
        # Reset performance metrics
        global _perf_metrics
        _perf_metrics = PerformanceMetrics()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache clear failed: {e}",
        )


@app.post("/cache/threshold", response_model=dict[str, Any])
async def set_threshold(request: ThresholdRequest) -> dict[str, Any]:
    """
    Update the cache distance threshold.

    Args:
        request: Threshold update request.

    Returns:
        Confirmation with new threshold value.
    """
    cache = get_cache()

    try:
        if not 0 <= request.threshold <= 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Threshold must be between 0 and 2 for cosine distance",
            )

        cache.set_threshold(request.threshold)
        return {
            "message": "Threshold updated",
            "threshold": request.threshold,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threshold update failed: {e}",
        )


@app.get("/cache/threshold", response_model=dict[str, float])
async def get_threshold() -> dict[str, float]:
    """Get the current cache distance threshold."""
    cache = get_cache()
    return {"threshold": cache.distance_threshold}


@app.get("/stats", response_model=dict[str, Any])
async def get_stats() -> dict[str, Any]:
    """Get cache statistics."""
    cache = get_cache()

    try:
        cache_stats = cache.get_stats()
        perf_stats = _perf_metrics.to_dict()

        return {
            "cache": cache_stats,
            "performance": perf_stats,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {e}",
        )


@app.get("/stats/reset", response_model=dict[str, str])
async def reset_stats() -> dict[str, str]:
    """Reset performance metrics."""
    global _perf_metrics
    _perf_metrics = PerformanceMetrics()
    return {"message": "Performance metrics reset"}


@app.post("/cache/embedding", response_model=dict[str, Any])
async def get_embedding(request: CacheCheckRequest) -> dict[str, Any]:
    """
    Get embedding for a prompt (useful for testing and understanding).

    Args:
        request: Request with prompt text.

    Returns:
        Embedding vector and metadata.
    """
    embedding_service = EmbeddingService()

    try:
        start_time = time.time()
        embedding = embedding_service.encode(request.prompt)
        encode_time_ms = (time.time() - start_time) * 1000

        return {
            "prompt": request.prompt,
            "embedding_dimension": len(embedding),
            "encode_time_ms": encode_time_ms,
            "model": embedding_service._model_name,
            # Return first 10 values for preview
            "embedding_preview": embedding[:10].tolist(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding: {e}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "semantic_cache.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
