"""Dependency injection configuration for FastAPI app.

Uses FastAPI's app.state pattern for storing service instances.

Pattern:
    - Services stored in app.state during lifespan
    - Dependency functions retrieve from request.app.state
    - Clean separation, no global mutable state
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from semantic_cache.handlers import CacheHandler
from semantic_cache.repositories import LocalEmbeddingProvider, RedisCacheRepository
from semantic_cache.services import CacheService


def get_cache_service(request: Request) -> CacheService:
    """Dependency injection for CacheService from app.state.

    Args:
        request: FastAPI Request object

    Returns:
        The CacheService instance from app.state

    Raises:
        RuntimeError: If service is not initialized
    """
    service = getattr(request.app.state, "cache_service", None)
    if service is None:
        raise RuntimeError("CacheService not initialized. Check lifespan setup.")
    return service


def get_handler(request: Request) -> CacheHandler:
    """Dependency injection for CacheHandler from app.state.

    Args:
        request: FastAPI Request object

    Returns:
        The CacheHandler instance from app.state

    Raises:
        RuntimeError: If handler is not initialized
    """
    handler = getattr(request.app.state, "cache_handler", None)
    if handler is None:
        raise RuntimeError("CacheHandler not initialized. Check lifespan setup.")
    return handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app.

    Initializes all layers and stores in app.state:
    1. Repository (data access) - created explicitly
    2. Service (business logic) - stored in app.state.cache_service
    3. Handler (HTTP endpoints) - stored in app.state.cache_handler

    Args:
        app: The FastAPI application instance

    Yields:
        None

    Cleanup:
        Removes all services from app.state on shutdown
    """
    # Initialize repositories
    embedding_provider = LocalEmbeddingProvider.create()
    repository = RedisCacheRepository.create(embedding_provider=embedding_provider)

    # Initialize services with explicit dependencies
    cache_service = CacheService.create(
        repository=repository,
        embedding_provider=embedding_provider,
    )
    cache_handler = CacheHandler(cache_service=cache_service)

    # Store in app.state (FastAPI pattern)
    app.state.cache_service = cache_service
    app.state.cache_handler = cache_handler
    app.state.embedding_provider = embedding_provider
    app.state.repository = repository

    print("✓ Cache service initialized")
    print(f"✓ Threshold: {cache_service.threshold}")
    print(f"✓ Health: {cache_service.is_healthy()}")

    yield

    # Cleanup - remove from app.state
    del app.state.cache_handler
    del app.state.cache_service
    del app.state.embedding_provider
    del app.state.repository
    print("✓ Cache service shut down")


# Type aliases for cleaner dependency injection
HandlerDep = Annotated[CacheHandler, Depends(get_handler)]
ServiceDep = Annotated[CacheService, Depends(get_cache_service)]
