# Semantic Cache Service

A semantic caching web service using Redis vector search and sentence-transformers. This project demonstrates how to build an intelligent caching system that understands semantic similarity, supporting multilingual queries (Indonesian & English).

## What is Semantic Caching?

Semantic caching goes beyond traditional exact-match caching by understanding the **meaning** of queries. Instead of requiring identical text, it finds semantically similar queries using vector embeddings, enabling:

- **70%+ reduction** in LLM API costs
- **80%+ faster** responses
- Cross-language matching (Indonesian query ↔ English cached response)

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│  FastAPI     │─────▶│   Redis     │
│             │◀─────│  Service     │◀─────│  Vector DB  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Sentence    │
                     │ Transformers │
                     │  Multilingual│
                     └──────────────┘
```

### Layered Architecture

The codebase follows clean architecture with clear separation of concerns:

```
┌──────────────────────────────────────────┐
│         API Layer (HTTP Routes)          │
├──────────────────────────────────────────┤
│       Handler Layer (HTTP Logic)         │
├──────────────────────────────────────────┤
│    Service Layer (Business Logic)        │
├──────────────────────────────────────────┤
│   Protocol Layer (Interface Contracts)   │
├──────────────────────────────────────────┤
│  Repository Layer (Data Access)          │
├──────────────────────────────────────────┤
│  Domain Layer (Entities, DTOs & Models)  │
└──────────────────────────────────────────┘
```

## Tech Stack

- **Python 3.11+**
- **FastAPI** - Web framework
- **Redis Stack** - Vector database and cache
- **sentence-transformers** - Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- **uv** - Fast Python package installer
- **ruff** - Fast Python linter & formatter
- **ty** - Fast Python type checker (Astral)

## Known Limitations & Future Strategies

### The Problem: Semantic Similarity ≠ Semantic Equivalence

Semantic caching relies on vector similarity, but similar prompts don't always have the same answer:

```
"What happened in 2024?" vs "What happened in 2025?"
→ ~95% similar by embedding distance
→ Completely different answers required
```

This is a fundamental challenge: embeddings capture meaning/topic similarity, but miss critical factual differences like dates, versions, or specific entities.

### Unimplemented Strategies

The following strategies could improve cache validity (not yet implemented):

#### 1. Critical Term Extraction
Extract important terms (dates, versions, numbers, names) and require exact match alongside semantic similarity:

```python
# Store with critical terms
{"prompt": "...", "critical_terms": ["2024", "python", "3.11"]}

# Check: semantic match + critical terms must match exactly
```

#### 2. Metadata Filtering (Partial Support)
Use existing metadata field for filtering during vector search:

```python
# Store with contextual metadata
{"prompt": "...", "metadata": {"year": 2024, "topic": "news"}}

# Check with filters (Redis supports hybrid vector + tag queries)
results = find_by_vector(vector, filters={"year": 2024})
```

#### 3. Two-Stage Validation
After finding semantically similar matches, perform secondary validation:

```python
def check(prompt):
    matches = find_similar(prompt)  # Stage 1: vector search
    for match in matches:
        if validate_match(prompt, match):  # Stage 2: validate critical terms
            return match
    return None
```

#### 4. Dynamic Threshold
Use stricter threshold for queries containing temporal or versioned content:

```python
def get_threshold(prompt):
    if has_temporal_terms(prompt):  # years, dates, versions
        return 0.05  # very strict
    return 0.15     # default
```

#### 5. Cache Partitioning
Partition cache by context (time period, domain, version) to prevent cross-contamination:

```
cache:2024:* → entries about 2024
cache:2025:* → entries about 2025
```

### Current Mitigations

- **TTL-based expiration**: Stale entries expire automatically (default: 7 days)
- **Threshold tuning**: Lower threshold reduces false positives (at cost of hit rate)
- **Metadata storage**: Infrastructure exists for metadata-based filtering

## Quick Start

### 1. Install Dependencies

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Start Redis

```bash
docker compose up -d
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults should work)
```

### 4. Run the API

```bash
uv run uvicorn semantic_cache.api.app:app --reload
```

The API will be available at `http://localhost:8000`

### 5. Try the Demo

```bash
uv run python scripts/demo.py
```

## API Endpoints

### Check Cache

```bash
curl -X POST http://localhost:8000/cache/check \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Apa itu semantic cache?"}'
```

**Response:**
```json
{
  "prompt": "Apa itu semantic cache?",
  "is_hit": true,
  "matches": [
    {
      "prompt": "Apa itu semantic cache?",
      "response": "Semantic cache adalah...",
      "vector_distance": 0.0000,
      "cosine_similarity": 1.0000,
      "cached_at": 1234567890.0
    }
  ],
  "lookup_time_ms": 5.23
}
```

### Store in Cache

```bash
curl -X POST http://localhost:8000/cache/store \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Apa itu semantic cache?",
    "response": "Semantic cache adalah sistem caching pintar...",
    "metadata": {"category": "concept"}
  }'
```

### Get Statistics

```bash
curl http://localhost:8000/cache/stats
```

**Response:**
```json
{
  "total_entries": 42,
  "index_name": "semantic_cache",
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_dimension": 384,
  "distance_threshold": 0.15
}
```

### Clear Cache

```bash
curl -X DELETE http://localhost:8000/cache/clear
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Project Structure

```
semantic-cache/
├── src/semantic_cache/
│   ├── __init__.py
│   ├── config.py                    # Settings (frozen dataclass), env loading
│   ├── api/
│   │   ├── app.py                   # FastAPI routes
│   │   └── dependencies.py          # DI container, lifespan manager
│   ├── dto/                         # Data Transfer Objects (API contracts)
│   │   ├── requests.py              # Pydantic request models
│   │   └── responses.py             # Pydantic response models
│   ├── entities/                    # Domain entities (frozen dataclasses)
│   │   ├── cache_entry.py           # CacheEntryEntity
│   │   └── cache_match.py           # CacheMatchEntity
│   ├── models/                      # Database models / external API contracts
│   │   └── *.py                     # DB schemas, third-party API models
│   ├── protocols/                   # Interface contracts (structural typing)
│   │   ├── cache_store.py           # CacheStore protocol
│   │   └── embedding_provider.py    # EmbeddingProvider protocol
│   ├── repositories/                # Data access implementations
│   │   ├── redis_repository.py      # RedisCacheRepository
│   │   └── local_embedding_provider.py
│   ├── services/
│   │   └── cache_service.py         # CacheService (business logic)
│   ├── handlers/
│   │   └── cache_handler.py         # CacheHandler (HTTP concerns)
│   └── utils/
│       └── evaluator.py             # Performance evaluation utilities
├── pyproject.toml                   # Dependencies and tool config
├── docker-compose.yml               # Redis service
├── CLAUDE.md                        # AI assistant instructions
└── README.md
```

## Key Design Patterns

### Protocol-Based Design
Services depend on protocols (interfaces), not concrete implementations. This allows swapping Redis for PostgreSQL/Qdrant without changing service code.

### Factory Methods
Classes use `create()` class methods for construction:
```python
cache_service = CacheService.create(repository, embedding_provider)
```

### Entity, DTO & Model Separation
- **Entities**: Frozen dataclasses for internal domain logic (immutable, no external deps)
- **DTOs**: Pydantic models for API request/response contracts (your API)
- **Models**: Database models or external API contracts (third-party APIs)

## Configuration

Environment variables (`.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `CACHE_DISTANCE_THRESHOLD` | `0.15` | Max distance for cache hit (0-2) |
| `CACHE_TTL` | `604800` | Cache entry TTL in seconds (7 days) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer model |
| `API_HOST` | `0.0.0.0` | API host |
| `API_PORT` | `8000` | API port |
| `CACHE_INDEX_NAME` | `semantic_cache` | Redis index name |

## Understanding Thresholds

The `distance_threshold` controls how similar queries must be to be considered a cache hit:

- **Lower (0.05-0.10)**: Very strict, high precision, low hit rate
- **Medium (0.10-0.15)**: Balanced, ~85-90% similarity (recommended)
- **Higher (0.15-0.25)**: Loose, high hit rate, risk of false positives

Note: For cosine distance, lower = more similar (0 = identical, 2 = opposite)

## Development

### Run Tests

```bash
uv run pytest
```

### Lint Code

```bash
uv run ruff check src/
```

### Format Code

```bash
uv run ruff format src/
```

### Type Check

```bash
uv run ty check src/
```

### Using Make

```bash
make test        # Run tests
make lint        # Lint code
make format      # Format code
make type-check  # Type check
make check       # Run all checks
make dev         # Start dev server
```

## Learning Concepts

### Cosine Similarity vs Distance

- **Cosine Similarity**: 1.0 = identical, 0.0 = orthogonal (unrelated)
- **Cosine Distance**: 0.0 = identical, 2.0 = opposite
- Relationship: `distance = 1 - similarity`

### Embeddings

Text is converted to 384-dimensional vectors using sentence-transformers. Semantically similar text has vectors that point in similar directions in vector space.

### HNSW Index

Redis uses Hierarchical Navigable Small World (HNSW) indexing for fast approximate nearest neighbor search - essential for production-scale caches.

## Performance Targets

Based on typical benchmarks:

| Metric | Target | Typical Range |
|--------|--------|---------------|
| Hit Rate | 60-70% | 40-80% |
| Latency Reduction | 70-80% | 50-90% |
| Cost Savings | 70%+ | 50-85% |
| Cache Lookup | <5ms | 1-10ms |

## Resources

- [Redis Vector Search](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
- [Semantic Caching Guide](https://redis.io/blog/what-is-semantic-caching/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## License

MIT
