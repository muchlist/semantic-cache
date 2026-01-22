"""Local sentence-transformers embedding provider.

This is the default embedding provider, using sentence-transformers
models running locally. No API calls required.
"""

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_cache.config import settings


class LocalEmbeddingProvider:
    """Local sentence-transformers implementation of EmbeddingProvider.

    This class satisfies the EmbeddingProvider protocol through structural
    typing - no explicit inheritance needed.

    Uses sentence-transformers models for multilingual text embeddings.
    Default model: paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the local embedding provider.

        Args:
            model_name: Name of the sentence-transformers model.
                       Defaults to settings.embedding_model.
        """
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @classmethod
    def create(cls, model_name: str | None = None) -> "LocalEmbeddingProvider":
        """Factory method to create LocalEmbeddingProvider with defaults.

        Args:
            model_name: Model name. If None, uses settings.

        Returns:
            Configured LocalEmbeddingProvider
        """
        return cls(model_name=model_name)

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            print(f"Loading embedding model: {self._model_name}")
            start_time = time.time()
            self._model = SentenceTransformer(self._model_name)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f}s")
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding vector dimension."""
        if self._dimension is None:
            # Get dimension by encoding a sample
            sample_embedding = self.model.encode(["test"], show_progress_bar=False)
            self._dimension = len(sample_embedding[0])
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    def encode(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: The text to encode

        Returns:
            The embedding vector as a list of floats
        """
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        # Handle both single string (returns array) and list input
        if isinstance(embedding, np.ndarray):
            if embedding.ndim == 1:
                return embedding.tolist()
            return embedding[0].tolist()
        return list(embedding)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def is_available(self) -> bool:
        """Check if the embedding provider is available.

        Returns:
            True if the model can be loaded, False otherwise
        """
        try:
            # Try to access the model property
            _ = self.model
            return True
        except Exception:
            return False
