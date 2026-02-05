# Embedding Models Guide

This guide explains the embedding models available for semantic caching and how to choose between them.

## Overview

This project uses **EmbeddingGemma** (Google's 768-dim multilingual embedding model) served via:
- **Ollama** (recommended) - Simple local serving
- **HuggingFace** (advanced) - Direct model loading with Matryoshka dimensions

## EmbeddingGemma Specifications

| Property | Value |
|----------|-------|
| **Model** | google/embeddinggemma-300m |
| **Parameters** | 308M |
| **Base Dimensions** | 768 |
| **Flexible Dimensions** | 768, 512, 256, 128 (via Matryoshka) |
| **Context Window** | 2,048 tokens (~1,500 words) |
| **Languages** | 100+ (optimized for multilingual) |
| **MTEB Score (Multilingual)** | 61.15 |
| **MTEB Score (English)** | 69.67 |
| **Size** | <200MB with quantization |
| **License** | Apache 2.0 (gated, requires acceptance) |

## Ollama vs HuggingFace Direct

| Feature | Ollama | HuggingFace Direct |
|---------|--------|-------------------|
| **Setup** | ⚡ 2 commands | ⚡ 1 extra step (auth) |
| **Authentication** | None needed | One-time login (`huggingface-cli login`) |
| **Dimensions** | Fixed (768 only) | ✅ Flexible (768/512/256/128) |
| **Model Management** | ✅ `ollama pull` | Auto-downloads on first use |
| **API** | ✅ HTTP (language-agnostic) | Python-only |
| **Memory** | ✅ Runs as service | Loaded in-process |
| **Multi-Model** | ✅ Easy switching | Reload required |
| **Performance** | ~Same | ~Same |
| **Best For** | No auth, simplicity | Dimension flexibility |

**Recommendation:**
- **Use Ollama** if: You want zero authentication, prefer external service
- **Use HuggingFace** if: You need Matryoshka dimensions (256/128) for storage savings

---

## Matryoshka Dimensions (HuggingFace Only)

### What is Matryoshka Representation Learning?

EmbeddingGemma is trained with MRL, allowing you to **truncate vectors** to smaller dimensions without retraining:

```python
# Full precision
embedding = provider.encode("text")  # 768 dims

# Truncate to 256 dims
compressed = embedding[:256]  # Still semantically meaningful!
```

### Dimension Trade-offs

| Dimension | Quality Loss | Storage Savings | Use Case |
|-----------|--------------|-----------------|----------|
| **768** (full) | 0% (baseline) | 0% | Production, maximum quality |
| **512** | ~5% | 33% smaller | Balanced quality/storage |
| **256** | ~10% | 66% smaller | Large-scale deployments |
| **128** | ~15% | 83% smaller | Archival, cold storage |

### Configuration

```python
# In dependencies.py

# Full precision (768 dims)
embedding_provider = GemmaEmbeddingProvider.create(
    model_name="google/embeddinggemma-300m",
    output_dimension=768
)

# Balanced (512 dims)
embedding_provider = GemmaEmbeddingProvider.create(
    output_dimension=512
)

# Storage-efficient (256 dims)
embedding_provider = GemmaEmbeddingProvider.create(
    output_dimension=256
)

# Maximum compression (128 dims)
embedding_provider = GemmaEmbeddingProvider.create(
    output_dimension=128
)
```

**Important:** Run `make cache-clear` when changing dimensions!

### When to Use Which Dimension

**768 (Full Precision)**
- ✅ Production deployments
- ✅ Maximum accuracy required
- ✅ Storage is not a concern
- ❌ Large-scale (millions of entries)

**512 (Balanced)**
- ✅ Good balance of quality/storage
- ✅ Most production use cases
- ✅ 33% storage savings worth it
- ❌ Need absolute maximum accuracy

**256 (Efficient)**
- ✅ Large deployments (100K+ entries)
- ✅ Cost-sensitive environments
- ✅ 66% storage savings significant
- ❌ Accuracy is critical

**128 (Maximum Compression)**
- ✅ Archival/cold storage
- ✅ Massive scale (millions of entries)
- ✅ 83% storage savings essential
- ❌ Primary production cache

### Storage Calculator

**Per 10,000 cache entries:**

| Dimension | Storage Size |
|-----------|--------------|
| 768 dims | ~30 MB |
| 512 dims | ~20 MB |
| 256 dims | ~10 MB |
| 128 dims | ~5 MB |

**Example:**
- 1 million entries at 768 dims = ~3 GB
- 1 million entries at 256 dims = ~1 GB (save 2 GB!)

---

## Alternative Models (Ollama Only)

Ollama supports multiple embedding models. You can switch by changing `model_name` in `dependencies.py`.

### Nomic Embed Text

```bash
ollama pull nomic-embed-text
```

**Specs:**
- Dimensions: 768
- Context: **8,192 tokens** (4x larger than Gemma!)
- Best for: Long documents

**Configuration:**
```python
embedding_provider = OllamaEmbeddingProvider.create(
    model_name="nomic-embed-text"
)
```

### MXBai Embed Large

```bash
ollama pull mxbai-embed-large
```

**Specs:**
- Dimensions: 1,024
- Context: 512 tokens
- Best for: Maximum accuracy

**Configuration:**
```python
embedding_provider = OllamaEmbeddingProvider.create(
    model_name="mxbai-embed-large"
)
```

### All-MiniLM

```bash
ollama pull all-minilm
```

**Specs:**
- Dimensions: 384
- Context: 256 tokens
- Size: 46 MB (very lightweight!)
- Best for: Low-resource environments

**Configuration:**
```python
embedding_provider = OllamaEmbeddingProvider.create(
    model_name="all-minilm"
)
```

**Remember:** Run `make cache-clear` when switching models!

---

## Performance Benchmarks

### Inference Speed (Apple M1)

**Single Query:**
| Model | Provider | Time | Relative |
|-------|----------|------|----------|
| EmbeddingGemma 768 | Ollama | ~100ms | 1.0x |
| EmbeddingGemma 768 | HuggingFace | ~95ms | 0.95x |
| EmbeddingGemma 256 | HuggingFace | ~90ms | 0.9x |
| Nomic Embed | Ollama | ~110ms | 1.1x |
| MXBai Embed | Ollama | ~120ms | 1.2x |
| All-MiniLM | Ollama | ~45ms | 0.45x |

**Batch (32 queries):**
| Model | Time | Per Query |
|-------|------|-----------|
| EmbeddingGemma 768 | ~1.5s | ~47ms |
| All-MiniLM | ~0.8s | ~25ms |

### Accuracy (MTEB Benchmark)

| Model | Multilingual | English | Code |
|-------|--------------|---------|------|
| EmbeddingGemma | **61.15** | **69.67** | **68.76** |
| Nomic Embed | ~58 | ~62 | N/A |
| All-MiniLM | ~52 | ~58 | N/A |

---

## Choosing a Model

### Decision Tree

```
Do you need maximum accuracy?
├─ YES → EmbeddingGemma (Ollama or HF)
└─ NO → Continue

Do you have long documents (>500 words)?
├─ YES → Nomic Embed (8K context)
└─ NO → Continue

Is storage a major concern?
├─ YES → EmbeddingGemma with HF (use 256 dims)
└─ NO → EmbeddingGemma with Ollama (768 dims)

Is setup simplicity critical?
├─ YES → Ollama (any model)
└─ NO → HuggingFace (for dimension flexibility)
```

### Use Case Recommendations

**Semantic Cache (Current Project)**
- **Recommended:** EmbeddingGemma via Ollama (768 dims)
- **Why:** Best balance of quality, simplicity, production-readiness

**Large-Scale Deployments (>100K entries)**
- **Recommended:** EmbeddingGemma via HuggingFace (256 dims)
- **Why:** 66% storage savings with only ~10% quality loss

**Long Document Caching**
- **Recommended:** Nomic Embed via Ollama
- **Why:** 8K context window handles full articles/conversations

**Low-Resource Environments**
- **Recommended:** All-MiniLM via Ollama
- **Why:** Smallest model, fastest inference

**Research/Experimentation**
- **Recommended:** HuggingFace direct
- **Why:** Full control over dimensions, model versions

---

## Migration Between Models

### General Steps

1. **Choose new model** (see recommendations above)
2. **Update dependencies.py** with new provider configuration
3. **Clear Redis index**: `make cache-clear`
4. **Restart service**: `make dev`
5. **Test**: `make demo`
6. **Re-tune threshold** in `.env` if needed

### Example: Switch to Nomic Embed

```bash
# 1. Pull model
ollama pull nomic-embed-text

# 2. Edit dependencies.py
# Change model_name to "nomic-embed-text"

# 3. Clear Redis
make cache-clear

# 4. Restart
make dev
```

### Example: Enable Matryoshka Compression

```bash
# 1. Authenticate with HuggingFace
huggingface-cli login

# 2. Edit dependencies.py
# Comment Ollama, uncomment HuggingFace with output_dimension=256

# 3. Clear Redis
make cache-clear

# 4. Restart
make dev
```

---

## FAQ

### Q: Why EmbeddingGemma over other models?

**A:** Superior multilingual support, large context window (2K tokens), and Matryoshka flexibility make it ideal for semantic caching across languages.

### Q: Can I mix dimensions in the same Redis index?

**A:** No. All vectors in a Redis index must have the same dimension. For hierarchical storage, use separate indexes.

### Q: How do I know which dimension to use?

**A:** Start with 768 (maximum quality). If storage becomes an issue, benchmark with 256 dims - most use cases see <10% quality loss.

### Q: Does Ollama support Matryoshka dimensions?

**A:** No. Ollama serves models with fixed output dimensions (768 for EmbeddingGemma). Use HuggingFace direct for dimension flexibility.

### Q: Which is faster: Ollama or HuggingFace?

**A:** HuggingFace direct is slightly faster (~5ms) due to no HTTP overhead. But Ollama's benefits (simplicity, no auth) usually outweigh this.

### Q: Can I use both Ollama AND HuggingFace?

**A:** Not simultaneously in the same service. You can switch by editing `dependencies.py` and running `make cache-clear`.

### Q: What if my cache queries are >2K tokens?

**A:** Use **Nomic Embed** (8K context) via Ollama. It handles much longer documents.

### Q: How do I optimize for cost?

**A:** Use HuggingFace with 256 dims for 66% storage savings, or switch to All-MiniLM (384 dims) if quality loss is acceptable.

---

## Resources

- [EmbeddingGemma Official Guide](https://ai.google.dev/gemma/docs/embeddinggemma)
- [EmbeddingGemma on HuggingFace](https://huggingface.co/google/embeddinggemma-300m)
- [EmbeddingGemma Paper](https://arxiv.org/pdf/2509.20354)
- [Ollama Models Library](https://ollama.com/library)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

---

## Next Steps

1. ✅ Understand model trade-offs
2. Choose provider: [QUICKSTART.md](QUICKSTART.md)
3. Production deployment: [ADVANCED.md](ADVANCED.md)
4. Benchmark with your data: `make demo`
5. Tune threshold: Edit `CACHE_DISTANCE_THRESHOLD` in `.env`
