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

## Tech Stack

- **Python 3.11+**
- **FastAPI** - Web framework
- **Redis Stack** - Vector database and cache
- **sentence-transformers** - Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- **uv** - Fast Python package installer
- **ruff** - Fast Python linter & formatter
- **ty** - Fast Python type checker (Astral)

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
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "cache": {
    "index_name": "semantic_cache",
    "total_entries": 42,
    "distance_threshold": 0.15,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_dimension": 384
  },
  "performance": {
    "total_queries": 100,
    "cache_hits": 72,
    "cache_misses": 28,
    "hit_rate": 0.72,
    "avg_lookup_time_ms": 4.5
  }
}
```

### Set Threshold

```bash
curl -X POST http://localhost:8000/cache/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.20}'
```

### Clear Cache

```bash
curl -X DELETE http://localhost:8000/cache
```

## Project Structure

```
semantic-cache/
├── src/semantic_cache/
│   ├── __init__.py
│   ├── config.py           # Settings and environment variables
│   ├── models.py           # Pydantic models and dataclasses
│   ├── embeddings.py       # Embedding service wrapper
│   ├── cache.py            # Core semantic cache implementation
│   ├── evaluator.py        # Performance evaluation utilities
│   └── api/
│       ├── __init__.py
│       └── app.py          # FastAPI application
├── tests/
│   ├── __init__.py
│   └── test_api.py         # API tests
├── scripts/
│   └── demo.py             # Demo script
├── pyproject.toml          # Dependencies and tool config
├── docker-compose.yml      # Redis service
└── README.md
```

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
