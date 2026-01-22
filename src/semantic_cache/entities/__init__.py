"""Domain entities for internal representation.

These are pure dataclasses (frozen) used internally by services
and repositories. They are NOT used for API contracts - use DTOs
from the dto package for that.

Entities should have:
- No JSON serialization logic
- No Pydantic validation
- No external dependencies
- Pure domain logic only
"""

from .cache_entry import CacheEntryEntity
from .cache_match import CacheMatchEntity

__all__ = ["CacheEntryEntity", "CacheMatchEntity"]
