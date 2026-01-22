"""Redis implementation of CacheStore.

This repository uses Redis Stack with vector search capabilities (HNSW index).
It's the default implementation and satisfies the CacheStore protocol.
"""

import json
import struct
import time
from typing import Any

import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

from semantic_cache.config import get_redis_client, settings
from semantic_cache.protocols import EmbeddingProvider


class RedisCacheRepository:
    """Redis implementation using HNSW vector index.

    This class satisfies the CacheStore protocol through structural
    typing - no explicit inheritance needed.

    Uses Redis Stack's vector search with:
    - HNSW (Hierarchical Navigable Small World) algorithm
    - COSINE distance metric
    - Configurable TTL for entries
    """

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        index_name: str | None = None,
        ttl: int | None = None,
    ) -> None:
        """Initialize the Redis cache repository.

        Args:
            redis_client: Redis client instance. If None, creates default.
            embedding_provider: Provider for getting vector dimension.
            index_name: Name of the Redis search index.
            ttl: Time-to-live for entries in seconds.
        """
        self._client = redis_client or get_redis_client()
        self._index_name = index_name or settings.cache_index_name
        self._ttl = ttl or settings.cache_ttl
        self._index: SearchIndex | None = None

        # Initialize the index
        self._ensure_index(embedding_provider)

    @classmethod
    def create(
        cls,
        embedding_provider: EmbeddingProvider | None = None,
        index_name: str | None = None,
        ttl: int | None = None,
    ) -> "RedisCacheRepository":
        """Factory method to create RedisCacheRepository with defaults.

        Args:
            embedding_provider: Provider for vector dimension.
            index_name: Redis index name. If None, uses settings.
            ttl: Entry TTL in seconds. If None, uses settings.

        Returns:
            Configured RedisCacheRepository
        """
        return cls(
            embedding_provider=embedding_provider,
            index_name=index_name,
            ttl=ttl,
        )

    def _ensure_index(self, embedding_provider: EmbeddingProvider | None) -> None:
        """Ensure the Redis vector index exists."""
        if self._index is not None:
            return

        # Get dimension from provider or use default
        if embedding_provider:
            dimension = embedding_provider.dimension
        else:
            # Default dimension for paraphrase-multilingual-MiniLM-L12-v2
            dimension = 384

        # Define the index schema
        index_schema = {
            "index": {
                "name": self._index_name,
                "prefix": f"{self._index_name}:",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "prompt", "type": "text", "attrs": {"weight": 1.0}},
                {"name": "response", "type": "text"},
                {
                    "name": "prompt_vector",
                    "type": "vector",
                    "attrs": {
                        "dims": dimension,
                        "algorithm": "HNSW",
                        "metric": "COSINE",
                    },
                },
                {"name": "timestamp", "type": "numeric"},
                {"name": "metadata", "type": "text"},
            ],
        }

        self._index = SearchIndex.from_dict(index_schema)
        self._index.set_client(self._client)

        # Create the index if it doesn't exist
        try:
            self._index.create(overwrite=False)
            print(f"Created new index: {self._index_name}")
        except Exception as e:
            if "already exists" in str(e) or "Index already exists" in str(e):
                print(f"Using existing index: {self._index_name}")
            else:
                raise

    def store(
        self,
        prompt: str,
        response: str,
        vector: list[float],
        metadata: dict,
        ttl: int,
    ) -> str:
        """Store a cache entry in Redis.

        Args:
            prompt: The original prompt text
            response: The cached response
            vector: The embedding vector for the prompt
            metadata: Optional metadata dictionary
            ttl: Time-to-live in seconds

        Returns:
            The storage key for the entry
        """
        # Convert vector to float32 bytes for Redis
        vector_bytes = struct.pack(f"{len(vector)}f", *vector)

        # Create entry key
        key = f"{self._index_name}:{int(time.time() * 1000)}"

        # Prepare data
        timestamp = time.time()
        metadata_str = json.dumps(metadata or {})

        # Store in Redis - vector as bytes, other fields as strings
        pipe = self._client.pipeline()
        pipe.hset(
            key,
            mapping={
                "prompt": prompt,
                "response": response,
                "prompt_vector": vector_bytes,
                "timestamp": str(timestamp),
                "metadata": metadata_str,
            },
        )
        pipe.expire(key, ttl)
        pipe.execute()

        return key

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
        if self._index is None:
            return []

        # Create vector query
        query = VectorQuery(
            vector=vector,
            vector_field_name="prompt_vector",
            return_fields=["prompt", "response", "timestamp", "metadata"],
            num_results=limit * 5,  # Get more results, then filter
        )

        # Execute search
        results = self._index.query(query)

        # Filter by distance threshold and convert to tuples
        matches = []
        for result in results:
            distance = float(result.get("vector_distance", 2.0))
            if distance <= threshold:
                # Parse metadata
                metadata_dict: dict[str, Any] | None = None
                if "metadata" in result:
                    try:
                        metadata_dict = json.loads(result["metadata"])
                    except json.JSONDecodeError:
                        metadata_dict = {"raw": result["metadata"]}

                # Get timestamp as float
                cached_at = float(result.get("timestamp", 0))

                matches.append(
                    (
                        result["prompt"],
                        result["response"],
                        distance,
                        cached_at,
                        metadata_dict,
                    )
                )

        # Sort by distance (closest first) and limit
        matches.sort(key=lambda m: m[2])
        return matches[:limit]

    def delete_by_key(self, key: str) -> bool:
        """Delete a specific entry by key.

        Args:
            key: The storage key to delete

        Returns:
            True if deleted, False otherwise
        """
        result: int = self._client.delete(key)  # type: ignore[assignment]
        return result > 0

    def delete_by_prompt(self, prompt: str) -> int:
        """Delete entries by exact prompt match.

        Args:
            prompt: The prompt to match

        Returns:
            Number of entries deleted
        """
        count = 0
        for key in self._client.scan_iter(match=f"{self._index_name}:*"):
            # Get the prompt field
            stored_prompt = self._client.hget(key, "prompt")
            if stored_prompt and isinstance(stored_prompt, bytes) and stored_prompt.decode() == prompt:
                if self._client.delete(key):
                    count += 1
        return count

    def clear_all(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries deleted
        """
        # Delete the index
        if self._index:
            self._index.delete(drop=True)

        # Recreate index
        self._index = None
        self._ensure_index(None)

        # Return count (approximate - we cleared the index)
        return -1  # Cannot get accurate count after deletion

    def count_all(self) -> int:
        """Count total entries in the cache.

        Returns:
            Total number of cached entries
        """
        count = 0
        for key in self._client.scan_iter(match=f"{self._index_name}:*"):
            count += 1
        return count

    def health_check(self) -> bool:
        """Check if Redis is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            result = self._client.ping()
            return bool(result)
        except Exception:
            return False

    def get_stats(self) -> dict:
        """Get repository statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "index_name": self._index_name,
            "total_entries": self.count_all(),
            "ttl": self._ttl,
        }

    @property
    def client(self) -> redis.Redis:
        """Get the Redis client."""
        return self._client
