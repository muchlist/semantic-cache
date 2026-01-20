import json
import struct
import time
from typing import Any

import numpy as np
import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

from semantic_cache.config import get_redis_client, settings
from semantic_cache.embeddings import EmbeddingService
from semantic_cache.models import CacheMatch, CacheResult


class SemanticCache:
    """Semantic cache using Redis vector search."""

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        embedding_service: EmbeddingService | None = None,
        distance_threshold: float | None = None,
        ttl: int | None = None,
        index_name: str | None = None,
    ) -> None:
        """
        Initialize the semantic cache.

        Args:
            redis_client: Redis client instance. If None, creates default.
            embedding_service: Embedding service. If None, creates default.
            distance_threshold: Maximum distance for cache hits. Defaults to settings.
            ttl: Time-to-live for cache entries in seconds. Defaults to settings.
            index_name: Name of the Redis search index. Defaults to settings.
        """
        self._client = redis_client or get_redis_client()
        self._embedding_service = embedding_service or EmbeddingService()
        self._distance_threshold = distance_threshold or settings.cache_distance_threshold
        self._ttl = ttl or settings.cache_ttl
        self._index_name = index_name or settings.cache_index_name
        self._index: SearchIndex | None = None

        # Initialize the index
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Ensure the Redis vector index exists."""
        if self._index is not None:
            return

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
                        "dims": self._embedding_service.dimension,
                        "algorithm": "HNSW",
                        "metric": "COSINE",
                    },
                },
                {"name": "timestamp", "type": "numeric"},
                {"name": "metadata", "type": "text"},
            ],
        }

        self._index = SearchIndex.from_dict(index_schema)

        # Set the Redis client
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

    def check(self, prompt: str, distance_threshold: float | None = None) -> CacheResult:
        """
        Check cache for semantically similar prompts.

        Args:
            prompt: The prompt to search for.
            distance_threshold: Override default distance threshold.

        Returns:
            CacheResult with any matches found.
        """
        threshold = distance_threshold or self._distance_threshold

        # Generate embedding for the prompt
        embedding = self._embedding_service.encode(prompt)
        embedding_list = embedding.tolist()

        # Create vector query
        query = VectorQuery(
            vector=embedding_list,
            vector_field_name="prompt_vector",
            return_fields=["prompt", "response", "timestamp", "metadata"],
            num_results=5,
        )

        # Execute search
        if self._index is None:
            return CacheResult(matches=[], prompt=prompt, is_hit=False)
        results = self._index.query(query)

        # Filter by distance threshold
        matches = []
        for result in results:
            # Result is a dict with vector_distance as string
            distance = float(result.get("vector_distance", 2.0))  # Default to max distance if missing
            if distance <= threshold:
                metadata: dict[str, Any] = {}
                if "metadata" in result:
                    try:
                        metadata = json.loads(result["metadata"])
                    except json.JSONDecodeError:
                        metadata = {"raw": result["metadata"]}

                matches.append(
                    CacheMatch(
                        prompt=result["prompt"],
                        response=result["response"],
                        vector_distance=distance,
                        cached_at=float(result["timestamp"]),
                        metadata=metadata,
                    )
                )

        # Sort matches by distance (closest first)
        matches.sort(key=lambda m: m.vector_distance)

        return CacheResult(
            matches=matches,
            prompt=prompt,
            is_hit=len(matches) > 0,
        )

    def store(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a prompt/response pair in the cache.

        Args:
            prompt: The prompt text.
            response: The response text.
            metadata: Optional metadata to store with the entry.

        Returns:
            The key of the stored entry.
        """
        # Generate embedding
        embedding = self._embedding_service.encode(prompt)
        # Convert to float32 bytes for Redis vector search (384 dims × 4 bytes = 1536 bytes)
        vector_bytes = struct.pack(f"{len(embedding)}f", *embedding.tolist())

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
        pipe.expire(key, self._ttl)
        pipe.execute()

        return key

    def clear(self) -> None:
        """Clear all entries from the cache."""
        # Delete the index
        if self._index:
            self._index.delete(drop=True)

        # Recreate index
        self._index = None
        self._ensure_index()

    def delete(self, key: str) -> bool:
        """
        Delete a specific cache entry.

        Args:
            key: The key of the entry to delete.

        Returns:
            True if the entry was deleted, False otherwise.
        """
        result: int = self._client.delete(key)  # type: ignore[assignment]
        return result > 0

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        # Count entries with our prefix
        count = 0
        for key in self._client.scan_iter(match=f"{self._index_name}:*"):
            count += 1

        return {
            "index_name": self._index_name,
            "total_entries": count,
            "distance_threshold": self._distance_threshold,
            "ttl": self._ttl,
            "embedding_model": self._embedding_service._model_name,
            "embedding_dimension": self._embedding_service.dimension,
        }

    def hydrate_from_pairs(
        self, pairs: list[tuple[str, str]], metadata: dict[str, Any] | None = None
    ) -> int:
        """
        Hydrate cache from a list of prompt/response pairs.

        Args:
            pairs: List of (prompt, response) tuples.
            metadata: Optional metadata to apply to all entries.

        Returns:
            Number of entries added.
        """
        count = 0
        for prompt, response in pairs:
            try:
                self.store(prompt, response, metadata)
                count += 1
            except Exception as e:
                print(f"Error storing pair: {e}")

        return count

    def set_threshold(self, threshold: float) -> None:
        """
        Update the distance threshold.

        Args:
            threshold: New threshold value.
        """
        self._distance_threshold = threshold

    @property
    def distance_threshold(self) -> float:
        """Get current distance threshold."""
        return self._distance_threshold

    @property
    def client(self) -> redis.Redis:
        """Get the Redis client."""
        return self._client
