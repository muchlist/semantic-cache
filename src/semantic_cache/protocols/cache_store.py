"""Cache storage protocol.

Defines the interface for any cache storage backend that can store and
retrieve vector embeddings for semantic similarity search.

Implementations can include:
- Redis Stack with vector search (default)
- PostgreSQL with pgvector
- Qdrant
- Pinecone
- Weaviate
- Any other vector database
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheStore(Protocol):
    """Protocol for cache storage backends.

    Similar to Go's interface pattern - any type that implements these
    methods satisfies the protocol, no explicit inheritance needed.

    Example:
        ```python
        from semantic_cache.repositories.protocols import CacheStore

        # Type check passes for any matching implementation
        repo: CacheStore = RedisCacheRepository()
        repo: CacheStore = PgVectorRepository(...)
        ```
    """

    def store(
        self,
        prompt: str,
        response: str,
        vector: list[float],
        metadata: dict,
        ttl: int,
    ) -> str:
        """Store a cache entry.

        Args:
            prompt: The original prompt text
            response: The cached response
            vector: The embedding vector for the prompt
            metadata: Optional metadata dictionary
            ttl: Time-to-live in seconds

        Returns:
            The storage key for the entry
        """
        ...

    def find_by_vector(
        self,
        vector: list[float],
        threshold: float,
        limit: int = 1,
    ) -> list[tuple[str, str, float, float, dict | None]]:
        """Find similar entries by vector similarity.

        Args:
            vector: The query embedding vector
            threshold: Maximum distance for matches
            limit: Maximum number of results to return

        Returns:
            List of tuples (prompt, response, distance, cached_at, metadata)
        """
        ...

    def delete_by_key(self, key: str) -> bool:
        """Delete a specific entry by key.

        Args:
            key: The storage key to delete

        Returns:
            True if deleted, False otherwise
        """
        ...

    def delete_by_prompt(self, prompt: str) -> int:
        """Delete entries by exact prompt match.

        Args:
            prompt: The prompt to match

        Returns:
            Number of entries deleted
        """
        ...

    def clear_all(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries deleted
        """
        ...

    def count_all(self) -> int:
        """Count total entries in the cache.

        Returns:
            Total number of cached entries
        """
        ...

    def health_check(self) -> bool:
        """Check if the repository is accessible.

        Returns:
            True if healthy, False otherwise
        """
        ...

    def get_stats(self) -> dict:
        """Get repository statistics.

        Returns:
            Dictionary with stats (implementation-specific)
        """
        ...
