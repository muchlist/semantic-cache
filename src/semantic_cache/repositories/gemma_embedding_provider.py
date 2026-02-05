"""Google EmbeddingGemma embedding provider.

EmbeddingGemma is Google's efficient embedding model based on Gemma 3,
designed for on-device and local deployment with strong multilingual support.

Key features:
- 308M parameters, <200MB RAM with quantization
- 2K token context window (16x larger than MiniLM)
- 100+ languages support
- Matryoshka Representation Learning (flexible dimensions: 768 → 128)
- MTEB score: 61.15 (multilingual), 69.67 (English)

Requires:
    pip install -U sentence-transformers
    # Optional: Latest transformers for best compatibility
    pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
"""

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_cache.config import settings


class GemmaEmbeddingProvider:
    """Google EmbeddingGemma implementation of EmbeddingProvider protocol.

    This class satisfies the EmbeddingProvider protocol through structural
    typing - no explicit inheritance needed.

    Uses sentence-transformers to load EmbeddingGemma models locally.
    Default model: google/embeddinggemma-300m (768 dimensions base)

    Supports Matryoshka Representation Learning (MRL) for flexible dimensions:
    - 768: Full precision (default)
    - 512: Balanced quality/performance
    - 256: Faster, lower storage
    - 128: Maximum efficiency

    Example:
        ```python
        # Full precision
        provider = GemmaEmbeddingProvider.create()

        # Compressed for storage efficiency
        provider = GemmaEmbeddingProvider.create(output_dimension=256)
        ```
    """

    def __init__(
        self,
        model_name: str | None = None,
        output_dimension: int | None = None,
    ) -> None:
        """Initialize the Gemma embedding provider.

        Args:
            model_name: Name of the EmbeddingGemma model.
                       Defaults to settings.embedding_model if it's a Gemma model.
            output_dimension: Output vector dimension (128, 256, 512, or 768).
                             Uses Matryoshka truncation for smaller dims.
                             Defaults to settings.embedding_output_dimension.
        """
        self._model_name = model_name or settings.embedding_model
        self._output_dimension = output_dimension or settings.embedding_output_dimension
        self._model: SentenceTransformer | None = None

        # Validate output dimension
        valid_dims = [128, 256, 512, 768]
        if self._output_dimension not in valid_dims:
            raise ValueError(
                f"output_dimension must be one of {valid_dims}, got {self._output_dimension}"
            )

    @classmethod
    def create(
        cls,
        model_name: str | None = None,
        output_dimension: int | None = None,
    ) -> "GemmaEmbeddingProvider":
        """Factory method to create GemmaEmbeddingProvider with defaults.

        Args:
            model_name: Model name. If None, uses settings.
            output_dimension: Output vector dimension. If None, uses settings.

        Returns:
            Configured GemmaEmbeddingProvider

        Example:
            ```python
            # Use defaults from settings
            provider = GemmaEmbeddingProvider.create()

            # Custom configuration
            provider = GemmaEmbeddingProvider.create(
                model_name="google/embeddinggemma-300m",
                output_dimension=256
            )
            ```
        """
        return cls(model_name=model_name, output_dimension=output_dimension)

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model.

        Returns:
            Loaded SentenceTransformer model
        """
        if self._model is None:
            print(f"Loading EmbeddingGemma model: {self._model_name}")
            print(f"Output dimension: {self._output_dimension}")
            start_time = time.time()
            self._model = SentenceTransformer(
                self._model_name,
                trust_remote_code=True,  # Required for some Gemma models
            )
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f}s")
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding vector dimension.

        Returns:
            The configured output dimension (128, 256, 512, or 768)
        """
        return self._output_dimension

    @property
    def model_name(self) -> str:
        """Get the model name/identifier.

        Returns:
            Model name (e.g., "google/embeddinggemma-300m")
        """
        return self._model_name

    def _truncate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Truncate embedding to output dimension using Matryoshka.

        EmbeddingGemma is trained with Matryoshka Representation Learning,
        allowing truncation to smaller dimensions while maintaining quality.

        Args:
            embedding: Full 768-dimensional embedding

        Returns:
            Truncated embedding of size self._output_dimension
        """
        if self._output_dimension == 768:
            return embedding
        return embedding[..., : self._output_dimension]

    def encode(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: The text to encode (supports up to 2K tokens)

        Returns:
            The embedding vector as a list of floats

        Example:
            ```python
            provider = GemmaEmbeddingProvider.create()
            embedding = provider.encode("Hello, world!")
            print(len(embedding))  # 768 or configured dimension
            ```
        """
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Handle both single string (returns array) and list input
        if isinstance(embedding, np.ndarray):
            if embedding.ndim == 1:
                truncated = self._truncate_embedding(embedding)
                return truncated.tolist()
            truncated = self._truncate_embedding(embedding[0])
            return truncated.tolist()
        return list(embedding)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors

        Example:
            ```python
            provider = GemmaEmbeddingProvider.create()
            embeddings = provider.encode_batch([
                "First text",
                "Second text",
                "Third text"
            ])
            print(len(embeddings))  # 3
            print(len(embeddings[0]))  # 768 or configured dimension
            ```
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Truncate all embeddings
        truncated = self._truncate_embedding(embeddings)
        return truncated.tolist()

    def is_available(self) -> bool:
        """Check if the embedding provider is available.

        Returns:
            True if the model can be loaded, False otherwise

        Example:
            ```python
            provider = GemmaEmbeddingProvider.create()
            if provider.is_available():
                print("EmbeddingGemma is ready!")
            ```
        """
        try:
            # Try to access the model property
            _ = self.model
            return True
        except Exception:
            return False
