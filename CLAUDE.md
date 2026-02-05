# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a semantic caching web service that uses Redis vector search (HNSW index) and sentence-transformers for multilingual (Indonesian/English) intelligent caching. Instead of exact-match caching, it finds semantically similar queries using vector embeddings.

### High-Level Architecture

```
Client → FastAPI Service → Redis Stack (Vector DB)
              ↓
    Sentence Transformers (Multilingual Embeddings)
```

**Core flow:**
1. Client sends prompt to `/cache/check`
2. Service generates embedding using sentence-transformers (384-dim vectors)
3. Redis performs vector search using HNSW index with COSINE distance
4. If distance ≤ threshold, returns cached response (cache hit)
5. On miss, client calls LLM, then stores result via `/cache/store`

## Commands

### Setup & Development
```bash
make setup          # Initial setup (install deps + start Redis)
make install        # Install dependencies via uv
make dev            # Start development server (uvicorn with --reload)
make redis-up       # Start Redis with docker compose
make redis-down     # Stop Redis
make demo           # Run demo script showing multilingual caching
```

### Testing & Quality
```bash
make test           # Run pytest
make test-cov       # Run tests with coverage report
make lint           # Run ruff linter
make format         # Format code with ruff
make type-check     # Run ty type checker (Astral's fast type checker)
make check          # Run all checks (lint + type-check)
make fix            # Auto-fix linting issues
```

### Cache Operations (requires API running)
```bash
make cache-clear    # Clear all cache entries
make cache-stats    # Show cache statistics
make cache-health   # Check cache/Redis health
make api-check      # Test /cache/check endpoint
make api-store      # Test /cache/store endpoint
make api-docs       # Open API docs in browser
```

### Utilities
```bash
make clean          # Remove cache/build artifacts
make clean-all      # Deep clean (including Redis data + venv)
make deps           # Update dependencies
```

## Code Architecture

### Layered Architecture

The project follows a clean layered architecture with clear separation of concerns:

```
┌──────────────────────────────────────────┐
│         API Layer (HTTP Routes)          │  api/app.py
├──────────────────────────────────────────┤
│       Handler Layer (HTTP Logic)         │  handlers/cache_handler.py
├──────────────────────────────────────────┤
│    Service Layer (Business Logic)        │  services/cache_service.py
├──────────────────────────────────────────┤
│   Protocol Layer (Interface Contracts)   │  protocols/
├──────────────────────────────────────────┤
│  Repository Layer (Data Access)          │  repositories/
├──────────────────────────────────────────┤
│  Domain Layer (Entities, DTOs & Models)  │  entities/, dto/, models/
└──────────────────────────────────────────┘
```

**Data Flow:** HTTP Request → Handler → Service → Repository → Redis/Embeddings

### Directory Structure

```
src/semantic_cache/
├── __init__.py
├── config.py                    # Settings (frozen dataclass), env loading
├── api/
│   ├── app.py                   # FastAPI routes
│   └── dependencies.py          # DI container, lifespan manager
├── dto/                         # Data Transfer Objects (API contracts)
│   ├── requests.py              # Pydantic request models
│   └── responses.py             # Pydantic response models
├── entities/                    # Domain entities (frozen dataclasses)
│   ├── cache_entry.py           # CacheEntryEntity
│   └── cache_match.py           # CacheMatchEntity
├── models/                      # Database models / external API contracts
│   └── *.py                     # DB schemas, third-party API models
├── protocols/                   # Interface contracts (structural typing)
│   ├── cache_store.py           # CacheStore protocol
│   └── embedding_provider.py    # EmbeddingProvider protocol
├── repositories/                # Data access implementations
│   ├── redis_repository.py      # RedisCacheRepository
│   ├── ollama_embedding_provider.py  # OllamaEmbeddingProvider
│   └── gemma_embedding_provider.py   # GemmaEmbeddingProvider
├── services/
│   └── cache_service.py         # CacheService (business logic)
├── handlers/
│   └── cache_handler.py         # CacheHandler (HTTP concerns)
└── utils/
    └── evaluator.py             # Performance evaluation utilities
```

### Key Design Patterns

#### 1. Protocol-Based Design (Structural Typing)
Services depend on protocols, not concrete implementations. Allows swapping Redis for PostgreSQL/Qdrant without changing service code.

```python
class CacheStore(Protocol):
    def store(...): ...
    def find_by_vector(...): ...

class CacheService:
    def __init__(self, repository: CacheStore, ...):  # Protocol, not Redis type
```

#### 2. Factory Methods
Classes use `create()` class methods for construction with sensible defaults:
- `CacheService.create(repository, embedding_provider, ...)`
- `RedisCacheRepository.create(embedding_provider, redis_url, ...)`
- `OllamaEmbeddingProvider.create(model_name, base_url)`
- `GemmaEmbeddingProvider.create(model_name, output_dimension)`

#### 3. Dependency Injection via app.state
Lifespan context manager initializes services and stores them in `app.state`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cache_service = CacheService.create(...)
    yield
    # cleanup
```

#### 4. Entity, DTO & Model Separation
- **Entities** (frozen dataclasses): Internal domain logic, immutable, no Pydantic
- **DTOs** (Pydantic models): Your API request/response contracts
- **Models**: Database models or external API contracts (third-party APIs)

Entities can have methods and properties:
```python
@dataclass(frozen=True)
class CacheMatchEntity:
    distance: float

    def is_exact_match(self) -> bool:
        return self.distance == 0.0

    @property
    def similarity_score(self) -> float:
        return 1.0 - (self.distance / 2.0)
```

#### 5. Lazy Model Loading
Embedding model loads only when first accessed:
```python
@property
def model(self) -> SentenceTransformer:
    if self._model is None:
        self._model = SentenceTransformer(self._model_name)
    return self._model
```

### Key Classes

| Class | Layer | Purpose |
|-------|-------|---------|
| `CacheHandler` | Handler | HTTP concerns, DTO↔Entity conversion |
| `CacheService` | Service | Business logic orchestration |
| `RedisCacheRepository` | Repository | Redis vector storage/search |
| `OllamaEmbeddingProvider` | Repository | Ollama-served embeddings (default) |
| `GemmaEmbeddingProvider` | Repository | HuggingFace direct embeddings (advanced) |
| `CacheStore` | Protocol | Interface for cache storage |
| `EmbeddingProvider` | Protocol | Interface for embedding generation |
| `CacheEntryEntity` | Entity | Internal cache entry representation |
| `CacheMatchEntity` | Entity | Internal search result representation |

### API Endpoints

| Method | Endpoint | Request DTO | Response DTO |
|--------|----------|-------------|--------------|
| POST | `/cache/check` | `CheckCacheRequest` | `CacheCheckResponse` |
| POST | `/cache/store` | `StoreCacheRequest` | `CacheStoreResponse` |
| GET | `/cache/stats` | - | `CacheStatsResponse` |
| DELETE | `/cache/clear` | - | `dict` |
| GET | `/health` | - | `dict` |

### Vector Index Schema

Redis hash with fields: `prompt`, `response`, `prompt_vector` (Float32 bytes), `timestamp`, `metadata` (JSON). Index name defaults to "semantic_cache" with prefix pattern `semantic_cache:*`. HNSW algorithm, COSINE metric, 384 dimensions.

## Configuration

Key environment variables (`.env`):
- `REDIS_URL`: Redis connection (default: `redis://localhost:6379`)
- `CACHE_DISTANCE_THRESHOLD`: Max distance for hit (0-2, default: 0.15)
- `CACHE_TTL`: Entry TTL in seconds (default: 604800 = 7 days)
- `EMBEDDING_MODEL`: Model name (supports multiple providers, see below)
- `EMBEDDING_OUTPUT_DIMENSION`: Output dimension for Gemma (128, 256, 512, or 768)
- `API_HOST` / `API_PORT`: Server binding
- `CACHE_INDEX_NAME`: Redis index name

### Embedding Models

Two embedding providers are available:

**OllamaEmbeddingProvider** (Default):
- Model: `embeddinggemma` (via Ollama)
- Dimensions: 768 (fixed)
- Context: 2K tokens
- Setup: `ollama pull embeddinggemma`
- Best for: Zero authentication, simple local setup, production-ready

**GemmaEmbeddingProvider** (Advanced):
- Model: `google/embeddinggemma-300m` (direct via sentence-transformers)
- Dimensions: 768, 512, 256, or 128 (flexible via Matryoshka)
- Context: 2K tokens
- Setup: Requires HuggingFace authentication (gated model)
- Best for: Dimension flexibility, Matryoshka compression for storage optimization

**Switching providers**: Edit `src/semantic_cache/api/dependencies.py` and comment/uncomment the desired provider.

**📖 Full comparison and migration guide**: See [docs/MODELS.md](docs/MODELS.md)

## Testing

Tests use FastAPI `TestClient` with pytest-asyncio. Tests may return 500/503 if Redis is not running. Run with `make test`.

## Important Notes

- Python 3.11+ required
- Uses `uv` for package management (not pip/poetry)
- Uses `ty` (Astral's fast type checker) instead of mypy
- Redis Stack (not plain Redis) required for vector search
- Multilingual support is core feature (Indonesian/English demo in `scripts/demo.py`)
- Threshold tuning critical: too low = few hits, too high = false positives
- Cosine distance: 0 = identical, 2 = opposite. Default threshold is 0.15
