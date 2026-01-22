"""FastAPI application using layered architecture.

This is the main FastAPI app file - kept thin and focused on:
- Route definitions
- App configuration
- Importing DI setup from dependencies.py

Architecture:
    HTTP Routes (Handler) -> Business Logic (Service) -> Data Access (Repository)
"""

from fastapi import APIRouter, FastAPI

from semantic_cache.api.dependencies import HandlerDep, lifespan
from semantic_cache.dto import (
    CacheCheckResponse,
    CacheStatsResponse,
    CacheStoreResponse,
    CheckCacheRequest,
    StoreCacheRequest,
)

# Create FastAPI app
app = FastAPI(
    title="Semantic Cache API",
    description="Multilingual semantic caching with Redis vector search (Layered Architecture)",
    version="2.0.0",
    lifespan=lifespan,
)

# Create router
router = APIRouter(prefix="/cache", tags=["cache"])


@router.post("/check", response_model=CacheCheckResponse)
async def check_cache(
    request: CheckCacheRequest,
    handler: HandlerDep,
) -> CacheCheckResponse:
    """Check cache for semantically similar prompt.

    Args:
        request: Check cache request with prompt and optional threshold
        handler: Injected cache handler

    Returns:
        Cache check response with match status and data
    """
    return await handler.check_cache(request)


@router.post("/store", response_model=CacheStoreResponse)
async def store_cache(
    request: StoreCacheRequest,
    handler: HandlerDep,
) -> CacheStoreResponse:
    """Store a prompt-response pair in cache.

    Args:
        request: Store cache request with prompt, response, and optional metadata
        handler: Injected cache handler

    Returns:
        Cache store response with storage confirmation
    """
    return await handler.store_cache(request)


@router.get("/stats", response_model=CacheStatsResponse)
async def get_stats(handler: HandlerDep) -> CacheStatsResponse:
    """Get cache statistics.

    Args:
        handler: Injected cache handler

    Returns:
        Cache statistics response
    """
    return await handler.get_stats()


@router.delete("/clear")
async def clear_cache(handler: HandlerDep) -> dict:
    """Clear all cache entries.

    Args:
        handler: Injected cache handler

    Returns:
        Clear operation result
    """
    return await handler.clear_cache()


@app.get("/health")
async def health_check(handler: HandlerDep) -> dict:
    """Health check endpoint.

    Args:
        handler: Injected cache handler

    Returns:
        Health status
    """
    return await handler.health_check()


# Include router
app.include_router(router)
