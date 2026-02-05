# Quick Start Guide

Get semantic cache running in 5 minutes with **Ollama** (recommended) or **HuggingFace** (advanced).

## Prerequisites

- Python 3.11+
- Docker (for Redis)
- macOS/Linux/Windows

## Choose Your Setup

### Option A: Ollama (Recommended for simplicity)

**Why Ollama?**
- ✅ No authentication required
- ✅ 2-command setup
- ✅ Automatic model management
- ✅ Production-ready

**Limitations:**
- ❌ Fixed 768 dimensions (no Matryoshka compression)

### Option B: HuggingFace Direct (Recommended for flexibility)

**Why HuggingFace?**
- ✅ Matryoshka dimensions (768/512/256/128)
- ✅ Flexible output dimensions
- ✅ Direct model control

**Limitations:**
- ❌ Requires one-time HuggingFace authentication (simple: just `huggingface-cli login`)

---

## Setup: Option A (Ollama)

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from https://ollama.com/download

### 2. Pull EmbeddingGemma

```bash
# Download model (~600MB, one-time)
ollama pull embeddinggemma

# Verify
ollama list
```

### 3. Install Project Dependencies

```bash
cd semantic-cache
uv sync
```

### 4. Start Redis

```bash
docker compose up -d
```

### 5. Configure Environment

```bash
cp .env.example .env
# Defaults work for Ollama - no changes needed!
```

### 6. Verify Configuration

Edit `src/semantic_cache/api/dependencies.py` (should already be set):

```python
# Ollama provider (already configured)
embedding_provider = OllamaEmbeddingProvider.create(
    model_name="embeddinggemma",
    base_url="http://localhost:11434"
)
```

### 7. Start the Service

```bash
make dev
```

You should see:
```
✓ Ollama provider initialized
✓ Cache service initialized
✓ Health: True
```

### 8. Test It

```bash
# In another terminal
make demo
```

**Done!** ✅

---

## Setup: Option B (HuggingFace Direct)

### 1. Create HuggingFace Account

1. Go to https://huggingface.co/join
2. Sign up (free)
3. Verify email

### 2. Request Model Access

1. Visit https://huggingface.co/google/embeddinggemma-300m
2. Click **"Agree and access repository"**
3. Accept terms (instant approval)

### 3. Get Access Token

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login (will prompt for token)
huggingface-cli login
```

Create token at: https://huggingface.co/settings/tokens
- Click "New token"
- Name: `semantic-cache`
- Type: **Read**
- Copy token (starts with `hf_...`)
- Paste when prompted

### 4. Install Project Dependencies

```bash
cd semantic-cache
uv sync
```

### 5. Start Redis

```bash
docker compose up -d
```

### 6. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```bash
EMBEDDING_MODEL=google/embeddinggemma-300m
EMBEDDING_OUTPUT_DIMENSION=768  # or 512, 256, 128
```

### 7. Update Dependencies

Edit `src/semantic_cache/api/dependencies.py`:

```python
# Comment out Ollama
# embedding_provider = OllamaEmbeddingProvider.create(...)

# Uncomment HuggingFace provider
embedding_provider = GemmaEmbeddingProvider.create(
    model_name="google/embeddinggemma-300m",
    output_dimension=768  # or 512, 256, 128 for compression
)
```

### 8. Clear Redis Index

```bash
# Required when switching providers
make cache-clear
```

### 9. Start the Service

```bash
make dev
```

You should see:
```
Loading EmbeddingGemma model: google/embeddinggemma-300m
Output dimension: 768
Model loaded in 15.34s
✓ Cache service initialized
```

### 10. Test It

```bash
make demo
```

**Done!** ✅

---

## Verification

### Check Ollama is Running

```bash
curl http://localhost:11434/api/version
# Should return: {"version":"..."}
```

### Check HuggingFace Auth

```bash
huggingface-cli whoami
# Should show your username
```

### Check API Health

```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "redis": "connected",
  "embedding_provider": "available"
}
```

### Test Cache Operations

**Store:**
```bash
curl -X POST http://localhost:8000/cache/store \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "response": "ML is a subset of AI that enables systems to learn from data.",
    "metadata": {}
  }'
```

**Check:**
```bash
curl -X POST http://localhost:8000/cache/check \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'
```

**Stats:**
```bash
curl http://localhost:8000/cache/stats
```

---

## Troubleshooting

### Ollama Issues

**Error: "connection refused"**

Ollama not running:
```bash
# Start Ollama
ollama serve

# Or check if running
ps aux | grep ollama
```

**Error: "model not found"**

Model not pulled:
```bash
ollama pull embeddinggemma
ollama list
```

**Error: "Ollama API error"**

Wrong base URL in dependencies.py:
```python
# Should be:
base_url="http://localhost:11434"  # default Ollama port
```

### HuggingFace Issues

**Error: "401 Client Error" or "Access restricted"**

Not authenticated or no license:
1. Go to https://huggingface.co/google/embeddinggemma-300m
2. Click "Agree and access repository"
3. Run `huggingface-cli login`

**Error: "Invalid token"**

Token expired or wrong:
1. Go to https://huggingface.co/settings/tokens
2. Delete old token
3. Create new token (Read access)
4. Run `huggingface-cli login` again

**Model downloads slowly**

First download is ~600MB - be patient. Model cached at `~/.cache/huggingface/hub/`

### Redis Issues

**Error: "Could not connect to Redis"**

Redis not running:
```bash
docker compose up -d
docker compose ps
```

**Error: "Dimension mismatch"**

Old index with wrong dimensions:
```bash
make cache-clear
make dev
```

### General Issues

**Health check fails**

```bash
# 1. Check Redis
docker compose ps

# 2. Check Ollama (if using)
curl http://localhost:11434/api/version

# 3. Check HF auth (if using)
huggingface-cli whoami

# 4. Clear Redis index
make cache-clear

# 5. Restart service
make dev
```

**API not starting**

Check logs for errors:
```bash
make dev
# Read startup logs carefully
```

---

## Switching Between Providers

### Ollama → HuggingFace

1. Authenticate with HuggingFace: `huggingface-cli login`
2. Edit `dependencies.py` - comment Ollama, uncomment HuggingFace
3. Clear Redis: `make cache-clear`
4. Restart: `make dev`

### HuggingFace → Ollama

1. Install Ollama: `brew install ollama`
2. Pull model: `ollama pull embeddinggemma`
3. Edit `dependencies.py` - comment HuggingFace, uncomment Ollama
4. Clear Redis: `make cache-clear`
5. Restart: `make dev`

**Important:** Always run `make cache-clear` when switching providers!

---

## Alternative Embedding Models (Ollama Only)

### Nomic Embed (8K context!)

```bash
ollama pull nomic-embed-text
```

Update `dependencies.py`:
```python
embedding_provider = OllamaEmbeddingProvider.create(
    model_name="nomic-embed-text"
)
```

### MXBai Embed (highest quality)

```bash
ollama pull mxbai-embed-large
```

Update `dependencies.py`:
```python
embedding_provider = OllamaEmbeddingProvider.create(
    model_name="mxbai-embed-large"
)
```

**Remember:** Run `make cache-clear` when changing models!

---

## Next Steps

1. ✅ Service is running
2. Read [MODELS.md](MODELS.md) - Understand model comparison
3. Read [ADVANCED.md](ADVANCED.md) - Production deployment
4. Tune threshold in `.env` - Adjust `CACHE_DISTANCE_THRESHOLD`
5. Monitor performance - Use `make cache-stats`

## Resources

- [Ollama Documentation](https://docs.ollama.com)
- [EmbeddingGemma on HuggingFace](https://huggingface.co/google/embeddinggemma-300m)
- [EmbeddingGemma Guide](https://ai.google.dev/gemma/docs/embeddinggemma)
- [Redis Vector Search](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
