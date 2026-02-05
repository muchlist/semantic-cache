"""HTTP handlers for cache operations.

Handlers convert between DTOs (API contracts) and service calls.
They handle HTTP concerns like status codes, validation, and error handling.
"""

import time

from fastapi import HTTPException, status

from semantic_cache.dto import (
    CacheCheckResponse,
    CacheMatchItem,
    CacheStatsResponse,
    CacheStoreResponse,
    CheckCacheRequest,
    StoreCacheRequest,
)
from semantic_cache.services import CacheService


class CacheHandler:
    """HTTP handlers for cache operations.

    This handler delegates business logic to CacheService
    and handles HTTP-specific concerns like:
    - Converting entities to DTOs
    - Setting appropriate status codes
    - Error handling and responses

    Example:
        ```python
        from semantic_cache.services import CacheService
        from semantic_cache.handlers import CacheHandler

        cache_service = CacheService.create()
        handler = CacheHandler(cache_service=cache_service)

        # Use in FastAPI route
        @app.post("/cache/check", response_model=CacheCheckResponse)
        async def check_cache(request: CheckCacheRequest):
            return await handler.check_cache(request)
        ```
    """

    def __init__(self, cache_service: CacheService) -> None:
        """Initialize the cache handler.

        Args:
            cache_service: The cache service for business logic (required).
        """
        self._cache = cache_service

    async def check_cache(self, request: CheckCacheRequest) -> CacheCheckResponse:
        """Handle POST /cache/check requests.

        Args:
            request: The check cache request DTO

        Returns:
            CacheCheckResponse with match status and data (legacy format)

        Raises:
            HTTPException: If an error occurs during cache check
        """
        try:
            start_time = time.time()

            # Delegate to service layer - get multiple matches
            matches = await self._cache.check_all(
                prompt=request.prompt,
                threshold=request.threshold,
                limit=5,  # Get up to 5 matches like old implementation
            )

            lookup_time_ms = (time.time() - start_time) * 1000

            # Convert entities to DTOs (legacy format)
            match_items = []
            for match in matches:
                match_items.append(
                    CacheMatchItem(
                        prompt=match.prompt,
                        response=match.response,
                        vector_distance=match.distance,
                        cosine_similarity=1.0 - match.distance,  # Convert to similarity
                        cached_at=match.cached_at,
                        metadata=match.metadata or {},  # Include stored metadata
                    )
                )

            return CacheCheckResponse(
                prompt=request.prompt,
                is_hit=len(matches) > 0,
                matches=match_items,
                lookup_time_ms=lookup_time_ms,
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to check cache: {e}",
            ) from e

    async def store_cache(self, request: StoreCacheRequest) -> CacheStoreResponse:
        """Handle POST /cache/store requests.

        Args:
            request: The store cache request DTO

        Returns:
            CacheStoreResponse with storage confirmation

        Raises:
            HTTPException: If an error occurs during storage
        """
        try:
            key = await self._cache.store(
                prompt=request.prompt,
                response=request.response,
                metadata=request.metadata,
            )

            return CacheStoreResponse(
                success=True,
                key=key,
                message="Entry stored successfully",
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store entry: {e}",
            ) from e

    async def get_stats(self) -> CacheStatsResponse:
        """Handle GET /cache/stats requests.

        Returns:
            CacheStatsResponse with cache statistics

        Raises:
            HTTPException: If an error occurs while fetching stats
        """
        try:
            stats = self._cache.get_stats()

            return CacheStatsResponse(
                total_entries=stats.get("total_entries", 0),
                index_name=stats.get("index_name", ""),
                threshold=stats.get("distance_threshold", 0.0),
                ttl_seconds=stats.get("ttl", 0),
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get stats: {e}",
            ) from e

    async def clear_cache(self) -> dict:
        """Handle DELETE /cache/clear requests.

        Returns:
            Dict with clear operation result
        """
        try:
            count = self._cache.clear()

            return {
                "success": True,
                "deleted_count": count,
                "message": "Cache cleared successfully",
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear cache: {e}",
            ) from e

    async def health_check(self) -> dict:
        """Handle GET /health requests.

        Returns:
            Dict with health status
        """
        is_healthy = await self._cache.is_healthy()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "cache_healthy": is_healthy,
        }
