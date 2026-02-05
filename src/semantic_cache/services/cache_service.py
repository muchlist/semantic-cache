"""Cache service for core business logic.

This service orchestrates cache operations by coordinating
the repository (data access) and embedding provider (vector generation).
"""

from semantic_cache.config import settings
from semantic_cache.entities import CacheMatchEntity
from semantic_cache.protocols import CacheStore, EmbeddingProvider


class CacheService:
    """Core cache orchestration service.

    This service depends on PROTOCOLS, not concrete implementations:
    - CacheStore: can be Redis, PostgreSQL, Qdrant, etc.
    - EmbeddingProvider: can be local, OpenAI, Cohere, etc.

    This enables easy swapping of implementations without changing
    the service code.

    Example:
        ```python
        from semantic_cache.services import CacheService

        # Create with defaults (Redis + local embeddings)
        cache = CacheService.create()

        # Or with custom parameters
        cache = CacheService.create(distance_threshold=0.2)

        # Or with custom implementations
        cache = CacheService(
            repository=QdrantRepository(...),
            embedding_provider=OpenAIEmbeddingProvider(...),
        )
        ```
    """

    def __init__(
        self,
        repository: CacheStore,
        embedding_provider: EmbeddingProvider,
        distance_threshold: float | None = None,
        ttl: int | None = None,
    ) -> None:
        """Initialize the cache service.

        Args:
            repository: Cache storage backend (required).
            embedding_provider: Embedding generation service (required).
            distance_threshold: Maximum distance for cache hits (0-2). Defaults to settings.
            ttl: Time-to-live for cache entries in seconds. Defaults to settings.
        """
        self._repository = repository
        self._embeddings = embedding_provider
        self._threshold = distance_threshold or settings.cache_distance_threshold
        self._ttl = ttl or settings.cache_ttl

    @classmethod
    def create(
        cls,
        repository: CacheStore,
        embedding_provider: EmbeddingProvider,
        distance_threshold: float | None = None,
        ttl: int | None = None,
    ) -> "CacheService":
        """Factory method to create CacheService with sensible defaults.

        Like dict.fromkeys() or Path.home() - this is an alternative constructor
        that provides default implementations.

        Args:
            repository: Cache storage backend (required).
            embedding_provider: Embedding generation service (required).
            distance_threshold: Max distance for cache hits. If None, uses settings.
            ttl: Time-to-live in seconds. If None, uses settings.

        Returns:
            Configured CacheService instance

        Example:
            ```python
            from semantic_cache.repositories import RedisCacheRepository, LocalEmbeddingProvider
            from semantic_cache.services import CacheService

            cache = CacheService.create(
                repository=RedisCacheRepository.create(),
                embedding_provider=LocalEmbeddingProvider.create(),
            )

            # Or with custom threshold
            cache = CacheService.create(
                repository=RedisCacheRepository.create(),
                embedding_provider=LocalEmbeddingProvider.create(),
                distance_threshold=0.2,
            )
            ```
        """
        return cls(
            repository=repository,
            embedding_provider=embedding_provider,
            distance_threshold=distance_threshold,
            ttl=ttl,
        )

    async def check(
        self,
        prompt: str,
        threshold: float | None = None,
    ) -> CacheMatchEntity | None:
        """Check cache for semantically similar prompt.

        Business logic:
        1. Generate embedding for the query prompt
        2. Query repository for similar vectors
        3. Filter results by distance threshold
        4. Return the best match or None

        Args:
            prompt: The prompt to search for
            threshold: Override default distance threshold

        Returns:
            CacheMatchEntity if found, None otherwise
        """
        threshold = threshold or self._threshold

        # Generate embedding for the query
        vector = await self._embeddings.encode(prompt)

        # Query repository for similar entries
        matches = self._repository.find_by_vector(
            vector=vector,
            threshold=threshold,
            limit=1,
        )

        # Return the best match if found
        if matches:
            prompt_str, response, distance, cached_at, metadata = matches[0]
            return CacheMatchEntity(
                prompt=prompt_str,
                response=response,
                distance=distance,
                cached_at=cached_at,
                metadata=metadata,
            )

        return None

    async def check_all(
        self,
        prompt: str,
        threshold: float | None = None,
        limit: int = 5,
    ) -> list[CacheMatchEntity]:
        """Check cache for all semantically similar prompts.

        Business logic:
        1. Generate embedding for the query prompt
        2. Query repository for similar vectors
        3. Return all matches (up to limit)

        Args:
            prompt: The prompt to search for
            threshold: Override default distance threshold
            limit: Maximum number of matches to return

        Returns:
            List of CacheMatchEntity (sorted by distance, closest first)
        """
        threshold = threshold or self._threshold

        # Generate embedding for the query
        vector = await self._embeddings.encode(prompt)

        # Query repository for similar entries
        matches = self._repository.find_by_vector(
            vector=vector,
            threshold=threshold,
            limit=limit,
        )

        # Convert tuples to entities
        result = []
        for prompt_str, response, distance, cached_at, metadata in matches:
            result.append(
                CacheMatchEntity(
                    prompt=prompt_str,
                    response=response,
                    distance=distance,
                    cached_at=cached_at,
                    metadata=metadata,
                )
            )

        return result

    async def store(
        self,
        prompt: str,
        response: str,
        metadata: dict | None = None,
    ) -> str:
        """Store a prompt-response pair in cache.

        Business logic:
        1. Generate embedding for the prompt
        2. Create domain entity
        3. Delegate to repository for storage

        Args:
            prompt: The original prompt text
            response: The LLM response to cache
            metadata: Optional metadata (model, tokens, cost, etc.)

        Returns:
            The storage key for the entry
        """
        # Generate embedding for the prompt
        vector = await self._embeddings.encode(prompt)

        # Store via repository
        key = self._repository.store(
            prompt=prompt,
            response=response,
            vector=vector,
            metadata=metadata or {},
            ttl=self._ttl,
        )

        return key

    def delete_by_prompt(self, prompt: str) -> int:
        """Delete cache entries by exact prompt match.

        Args:
            prompt: The prompt to match

        Returns:
            Number of entries deleted
        """
        return self._repository.delete_by_prompt(prompt)

    def delete_by_key(self, key: str) -> bool:
        """Delete a specific cache entry by key.

        Args:
            key: The storage key to delete

        Returns:
            True if deleted, False otherwise
        """
        return self._repository.delete_by_key(key)

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        return self._repository.clear_all()

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = self._repository.get_stats()
        stats["distance_threshold"] = self._threshold
        stats["embedding_model"] = self._embeddings.model_name
        stats["embedding_dimension"] = self._embeddings.dimension
        return stats

    async def is_healthy(self) -> bool:
        """Check if cache is healthy.

        Returns:
            True if both repository and embeddings are healthy
        """
        repo_healthy = self._repository.health_check()
        embeddings_healthy = await self._embeddings.is_available()
        return repo_healthy and embeddings_healthy

    def set_threshold(self, threshold: float) -> None:
        """Update the similarity threshold.

        Args:
            threshold: New threshold value (0-2, lower = more strict)
        """
        if not 0 <= threshold <= 2:
            raise ValueError("Threshold must be between 0 and 2")
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Get current distance threshold."""
        return self._threshold

    @property
    def repository(self) -> CacheStore:
        """Get the underlying repository (for testing)."""
        return self._repository

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get the underlying embedding provider (for testing)."""
        return self._embeddings
