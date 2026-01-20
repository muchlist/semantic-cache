import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel


@dataclass
class CacheMatch:
    """A single cache match result."""

    prompt: str
    response: str
    vector_distance: float
    cached_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cosine_similarity(self) -> float:
        """Convert cosine distance to similarity (1 - distance)."""
        return max(0.0, 1.0 - self.vector_distance)

    @property
    def cached_at_datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.cached_at)


@dataclass
class CacheResult:
    """Result from a cache check operation."""

    matches: list[CacheMatch]
    prompt: str
    is_hit: bool

    @property
    def best_match(self) -> CacheMatch | None:
        """Get the best (closest) match, or None if no matches."""
        return self.matches[0] if self.matches else None


class CacheEntry(BaseModel):
    """Request/response entry for cache storage."""

    prompt: str
    response: str
    metadata: dict[str, Any] = {}

    model_config = {"extra": "allow"}


class CacheStats(BaseModel):
    """Cache statistics."""

    total_entries: int
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_lookup_time_ms: float

    model_config = {"extra": "allow"}


@dataclass
class PerformanceMetrics:
    """Track performance metrics for cache operations."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_lookup_time_ms: float = 0.0
    total_llm_time_ms: float = 0.0
    llm_calls: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def avg_lookup_time_ms(self) -> float:
        """Calculate average lookup time."""
        if self.total_queries == 0:
            return 0.0
        return self.total_lookup_time_ms / self.total_queries

    def record_hit(self, lookup_time_ms: float) -> None:
        """Record a cache hit."""
        self.total_queries += 1
        self.cache_hits += 1
        self.total_lookup_time_ms += lookup_time_ms

    def record_miss(self, lookup_time_ms: float) -> None:
        """Record a cache miss."""
        self.total_queries += 1
        self.cache_misses += 1
        self.total_lookup_time_ms += lookup_time_ms

    def record_llm_call(self, duration_ms: float) -> None:
        """Record an LLM API call."""
        self.llm_calls += 1
        self.total_llm_time_ms += duration_ms

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary."""
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "avg_lookup_time_ms": self.avg_lookup_time_ms,
            "total_llm_time_ms": self.total_llm_time_ms,
            "llm_calls": self.llm_calls,
        }


class LLMCallRecord(BaseModel):
    """Record of an LLM API call for cost tracking."""

    model: str
    prompt: str
    response: str
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    model_config = {"extra": "allow"}
