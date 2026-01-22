"""Handler layer for HTTP endpoints.

This layer contains the HTTP request/response handlers.
Handlers depend on services (business logic), not directly on repositories.

Architecture:
    Handler -> Service -> Repository
    (HTTP)  -> (Business) -> (Data Access)
"""

from .cache_handler import CacheHandler

__all__ = [
    "CacheHandler",
]
