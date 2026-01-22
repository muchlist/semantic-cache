"""Embedding provider protocol.

Defines the interface for any embedding generation service that can
convert text to vector embeddings.

Implementations can include:
- sentence-transformers (local, default)
- OpenAI embeddings (API)
- Cohere embeddings (API)
- HuggingFace inference API
- Custom embedding services
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding generation services.

    Similar to Go's interface pattern - any type that implements these
    methods satisfies the protocol, no explicit inheritance needed.

    Example:
        ```python
        from semantic_cache.repositories.protocols import EmbeddingProvider

        # Type check passes for any matching implementation
        provider: EmbeddingProvider = LocalEmbeddingProvider()
        provider: EmbeddingProvider = OpenAIEmbeddingProvider(...)
        ```
    """

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors.

        Returns:
            The vector dimension (e.g., 384 for paraphrase-multilingual-MiniLM-L12-v2)
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the name/identifier of the model.

        Returns:
            Model name or identifier
        """
        ...

    def encode(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: The text to encode

        Returns:
            The embedding vector as a list of floats
        """
        ...

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        ...

    def is_available(self) -> bool:
        """Check if the embedding provider is available.

        Returns:
            True if available, False otherwise
        """
        ...
