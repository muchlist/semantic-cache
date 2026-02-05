"""Ollama-based embedding provider.

Uses Ollama's local API to generate embeddings. Ollama serves models locally
without requiring HuggingFace authentication or downloading models manually.

Requirements:
    - Ollama installed: https://ollama.com
    - Model pulled: `ollama pull embeddinggemma`
    - Ollama running: `ollama serve` (usually runs automatically)

Key features:
- No HuggingFace authentication needed
- Simple HTTP API
- Automatic model download on first use
- Local-only, no external API calls

Models available:
- embeddinggemma (308M params, 768 dims, 2K context)
- embeddinggemma:300m (alias for above)
- nomic-embed-text (137M params, 768 dims)
- mxbai-embed-large (335M params, 1024 dims)
- all-minilm (22M params, 384 dims)
"""

import httpx

from semantic_cache.config import settings


class OllamaEmbeddingProvider:
    """Ollama-based implementation of EmbeddingProvider protocol.

    This class satisfies the EmbeddingProvider protocol through structural
    typing - no explicit inheritance needed.

    Uses Ollama's local API to generate embeddings. The API endpoint is
    http://localhost:11434/api/embed by default.

    Example:
        ```python
        # Create provider (will use Ollama)
        provider = OllamaEmbeddingProvider.create(
            model_name="embeddinggemma",
            base_url="http://localhost:11434"
        )

        # Generate embedding
        embedding = provider.encode("Hello, world!")
        print(len(embedding))  # 768
        ```
    """

    # Known model dimensions (for common models)
    MODEL_DIMENSIONS = {
        "embeddinggemma": 768,
        "embeddinggemma:300m": 768,
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "all-minilm:l6-v2": 384,
    }

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the Ollama embedding provider.

        Args:
            model_name: Name of the Ollama model.
                       Defaults to settings.embedding_model.
            base_url: Ollama API base URL.
                     Defaults to settings.ollama_base_url.
            timeout: Request timeout in seconds.
        """
        self._model_name = model_name or settings.embedding_model
        self._base_url = base_url or settings.ollama_base_url
        self._timeout = timeout
        self._dimension: int | None = None
        self._client = httpx.Client(timeout=self._timeout)

    @classmethod
    def create(
        cls,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> "OllamaEmbeddingProvider":
        """Factory method to create OllamaEmbeddingProvider with defaults.

        Args:
            model_name: Model name. If None, uses settings.
            base_url: Ollama API URL. If None, uses settings.

        Returns:
            Configured OllamaEmbeddingProvider

        Example:
            ```python
            # Use defaults from settings
            provider = OllamaEmbeddingProvider.create()

            # Custom configuration
            provider = OllamaEmbeddingProvider.create(
                model_name="embeddinggemma",
                base_url="http://localhost:11434"
            )
            ```
        """
        return cls(model_name=model_name, base_url=base_url)

    @property
    def dimension(self) -> int:
        """Get the embedding vector dimension.

        Returns:
            The vector dimension for the model

        Note:
            For known models, returns cached dimension.
            For unknown models, makes an API call to detect dimension.
        """
        if self._dimension is None:
            # Try known dimensions first
            if self._model_name in self.MODEL_DIMENSIONS:
                self._dimension = self.MODEL_DIMENSIONS[self._model_name]
            else:
                # Detect by encoding a sample
                sample_embedding = self.encode("test")
                self._dimension = len(sample_embedding)
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get the model name/identifier.

        Returns:
            Model name (e.g., "embeddinggemma")
        """
        return self._model_name

    def encode(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: The text to encode

        Returns:
            The embedding vector as a list of floats

        Raises:
            httpx.HTTPError: If Ollama API request fails
            ValueError: If response format is invalid

        Example:
            ```python
            provider = OllamaEmbeddingProvider.create()
            embedding = provider.encode("Hello, world!")
            print(len(embedding))  # 768
            ```
        """
        url = f"{self._base_url}/api/embed"
        payload = {
            "model": self._model_name,
            "input": text,
        }

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Ollama returns {"embeddings": [[...]]} for single input
            if "embeddings" in data and len(data["embeddings"]) > 0:
                return data["embeddings"][0]

            # Fallback: try "embedding" (singular)
            if "embedding" in data:
                return data["embedding"]

            raise ValueError(f"Unexpected response format: {data}")

        except httpx.HTTPError as e:
            error_msg = f"Ollama API error: {e}"
            if "connection refused" in str(e).lower():
                error_msg += "\n  → Is Ollama running? Try: ollama serve"
            elif "model" in str(e).lower() and "not found" in str(e).lower():
                error_msg += f"\n  → Model not found. Try: ollama pull {self._model_name}"
            raise RuntimeError(error_msg) from e

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding (Ollama handles batching internally)

        Returns:
            List of embedding vectors

        Example:
            ```python
            provider = OllamaEmbeddingProvider.create()
            embeddings = provider.encode_batch([
                "First text",
                "Second text",
                "Third text"
            ])
            print(len(embeddings))  # 3
            print(len(embeddings[0]))  # 768
            ```
        """
        url = f"{self._base_url}/api/embed"
        payload = {
            "model": self._model_name,
            "input": texts,  # Ollama accepts array of strings
        }

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            if "embeddings" in data:
                return data["embeddings"]

            raise ValueError(f"Unexpected response format: {data}")

        except httpx.HTTPError as e:
            error_msg = f"Ollama API error: {e}"
            if "connection refused" in str(e).lower():
                error_msg += "\n  → Is Ollama running? Try: ollama serve"
            elif "model" in str(e).lower() and "not found" in str(e).lower():
                error_msg += f"\n  → Model not found. Try: ollama pull {self._model_name}"
            raise RuntimeError(error_msg) from e

    def is_available(self) -> bool:
        """Check if the embedding provider is available.

        Returns:
            True if Ollama is running and model is available, False otherwise

        Example:
            ```python
            provider = OllamaEmbeddingProvider.create()
            if provider.is_available():
                print("Ollama is ready!")
            else:
                print("Ollama is not available. Run: ollama serve")
            ```
        """
        try:
            # Try to generate a test embedding
            _ = self.encode("test")
            return True
        except Exception:
            return False

    def __del__(self):
        """Cleanup: close HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()
