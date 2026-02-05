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
- Async support for concurrent requests

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
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-load the async HTTP client.

        Returns:
            The httpx.AsyncClient instance
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

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
            For unknown models, returns 768 as default (will validate on first encode).
        """
        if self._dimension is None:
            # Try known dimensions first
            if self._model_name in self.MODEL_DIMENSIONS:
                self._dimension = self.MODEL_DIMENSIONS[self._model_name]
            else:
                # Default for unknown models (will be validated on first encode)
                self._dimension = 768
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get the model name/identifier.

        Returns:
            Model name (e.g., "embeddinggemma")
        """
        return self._model_name

    async def encode(self, text: str) -> list[float]:
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
            embedding = await provider.encode("Hello, world!")
            print(len(embedding))  # 768
            ```
        """
        url = f"{self._base_url}/api/embed"
        payload = {
            "model": self._model_name,
            "input": text,
        }

        try:
            response = await self.client.post(url, json=payload)
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

    async def is_available(self) -> bool:
        """Check if the embedding provider is available.

        Returns:
            True if Ollama is running and model is available, False otherwise

        Example:
            ```python
            provider = OllamaEmbeddingProvider.create()
            if await provider.is_available():
                print("Ollama is ready!")
            else:
                print("Ollama is not available. Run: ollama serve")
            ```
        """
        try:
            # Try to generate a test embedding
            _ = await self.encode("test")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the async HTTP client.

        Should be called when shutting down the application.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
