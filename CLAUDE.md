# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a semantic caching web service that uses Redis vector search (HNSW index) and sentence-transformers for multilingual (Indonesian/English) intelligent caching. Instead of exact-match caching, it finds semantically similar queries using vector embeddings.

### Architecture

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

### Core Modules (`src/semantic_cache/`)

- **`config.py`**: Settings via frozen dataclass, loads from `.env`, provides `get_redis_client()` and global `settings` instance
- **`embeddings.py`**: `EmbeddingService` class wrapping sentence-transformers with lazy model loading and batch encoding support
- **`models.py`**: Dataclasses (`CacheMatch`, `CacheResult`, `PerformanceMetrics`) and Pydantic models (`CacheEntry`, `CacheStats`)
- **`cache.py`**: `SemanticCache` class - core implementation using `redisvl.SearchIndex` and `VectorQuery` for HNSW vector search
- **`evaluator.py`**: `CacheEvaluator` for threshold tuning and performance evaluation, `SimplePerfEval` for cost tracking
- **`api/app.py`**: FastAPI application with global `SemanticCache` singleton and `PerformanceMetrics` tracker

### Key Design Patterns

1. **Global singleton cache**: `get_cache()` in `api/app.py` creates/returns global `SemanticCache` instance
2. **Lazy model loading**: `EmbeddingService.model` property loads sentence-transformer on first use
3. **Redis index auto-creation**: `SemanticCache._ensure_index()` creates HNSW index if missing, connects if exists
4. **Cosine distance matching**: Lower distance = more similar (0 = identical, 2 = opposite). Threshold default is 0.15

### Vector Index Schema

Redis hash with fields: `prompt`, `response`, `prompt_vector` (JSON array), `timestamp`, `metadata`. Index name defaults to "semantic_cache" with prefix pattern `semantic_cache:*`. HNSW algorithm, COSINE metric, 384 dimensions.

## Configuration

Key environment variables (`.env`):
- `REDIS_URL`: Redis connection (default: `redis://localhost:6379`)
- `CACHE_DISTANCE_THRESHOLD`: Max distance for hit (0-2, default: 0.15)
- `CACHE_TTL`: Entry TTL in seconds (default: 604800 = 7 days)
- `EMBEDDING_MODEL`: Model name (default: `paraphrase-multilingual-MiniLM-L12-v2`)

## Testing

Tests in `tests/test_api.py` use FastAPI `TestClient`. Tests may return 500/503 if Redis is not running. Run with `make test`.

## Important Notes

- Python 3.11+ required
- Uses `uv` for package management (not pip/poetry)
- Uses `ty` (Astral's fast type checker) instead of mypy
- Redis Stack (not plain Redis) required for vector search
- Multilingual support is core feature (Indonesian/English demo in `scripts/demo.py`)
- Threshold tuning critical: too low = few hits, too high = false positives
