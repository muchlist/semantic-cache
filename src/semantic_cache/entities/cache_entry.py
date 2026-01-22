"""Cache entry domain entity."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CacheEntryEntity:
    """Domain entity for a cached prompt-response pair.

    This is an internal representation used by services and repositories.
    For API contracts, use the DTO classes from the dto package.

    Attributes:
        prompt: The original user prompt
        response: The cached LLM response
        prompt_vector: The embedding vector for the prompt
        timestamp: When this entry was created
        metadata: Optional additional data (e.g., model used, tokens, etc.)
    """

    prompt: str
    response: str
    prompt_vector: list[float]
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
