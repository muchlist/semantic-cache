import time
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_cache.config import settings


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to settings.embedding_model.
        """
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

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
        """Get the embedding dimension."""
        if self._dimension is None:
            # Get dimension by encoding a sample
            sample_embedding = self.model.encode(["test"], show_progress_bar=False)
            self._dimension = len(sample_embedding[0])
        return self._dimension

    def encode(self, text: str | list[str]) -> np.ndarray:
        """
        Encode text to embedding vector.

        Args:
            text: Single text string or list of text strings.

        Returns:
            Embedding vector(s) as numpy array.
        """
        return self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple texts in batches for efficiency.

        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.

        Returns:
            Embedding vectors as numpy array.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service instance."""
    return EmbeddingService()
