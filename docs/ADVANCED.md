# Advanced Topics

Production deployment, Redis migration, troubleshooting, and optimization.

## Table of Contents

- [Redis Index Migration](#redis-index-migration)
- [Production Deployment](#production-deployment)
- [Security Best Practices](#security-best-practices)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## Redis Index Migration

### Why Clear the Index?

Redis vector indexes have **fixed dimensions**. When you:
- Switch embedding models (Gemma → Nomic)
- Change Matryoshka dimensions (768 → 256)
- Switch providers (Ollama → HuggingFace)

The old index will cause dimension mismatch errors.

### Quick Migration

**Using Make (Easiest):**
```bash
make cache-clear
```

**Using Redis CLI:**
```bash
redis-cli FT.DROPINDEX semantic_cache DD
```

**Using Python:**
```python
from semantic_cache.config import get_redis_client

client = get_redis_client()
client.execute_command("FT.DROPINDEX", "semantic_cache", "DD")
```

### Complete Migration Workflow

**Example: Switch from 768 → 256 dimensions**

```bash
# 1. Stop the API
Ctrl+C

# 2. Update dependencies.py
# Change: output_dimension=768 → output_dimension=256

# 3. Clear Redis index
make cache-clear

# 4. Restart API
make dev

# 5. Verify logs
# Look for: "Created new index: semantic_cache"
```

### What Gets Deleted?

| Command | Index | Data |
|---------|-------|------|
| `FT.DROPINDEX semantic_cache` | ✅ Deleted | ❌ Kept (unsearchable) |
| `FT.DROPINDEX semantic_cache DD` | ✅ Deleted | ✅ Deleted |

**`make cache-clear` uses the `DD` flag** - deletes everything.

### Check Current Index

```bash
redis-cli FT.INFO semantic_cache
```

Look for the `VECTOR` field to see current dimensions.

---

## Production Deployment

### Docker

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  # Ollama (if using)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

  # Semantic Cache API
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - EMBEDDING_MODEL=embeddinggemma
      # For HuggingFace (if using):
      # - HF_TOKEN=${HF_TOKEN}
      # - EMBEDDING_OUTPUT_DIMENSION=768
    depends_on:
      - redis
      - ollama  # Remove if using HuggingFace
    restart: unless-stopped

  # Redis Stack
  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  ollama_data:
  redis_data:
```

**Pull Ollama model:**
```bash
docker-compose exec ollama ollama pull embeddinggemma
```

**Start services:**
```bash
docker-compose up -d
```

### Kubernetes

**ollama-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        volumeMounts:
        - name: models
          mountPath: /root/.ollama
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: ollama-models

---
apiVersion: v1
kind: Service
metadata:
  name: ollama
spec:
  ports:
  - port: 11434
  selector:
    app: ollama
```

**api-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-cache
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-cache
  template:
    metadata:
      labels:
        app: semantic-cache
    spec:
      containers:
      - name: api
        image: your-registry/semantic-cache:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis:6379"
        - name: EMBEDDING_MODEL
          value: "embeddinggemma"
        # For HuggingFace:
        # - name: HF_TOKEN
        #   valueFrom:
        #     secretKeyRef:
        #       name: huggingface-token
        #       key: HF_TOKEN
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: semantic-cache
spec:
  type: LoadBalancer
  ports:
  - port: 8000
  selector:
    app: semantic-cache
```

**secret.yaml (for HuggingFace):**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: huggingface-token
type: Opaque
stringData:
  HF_TOKEN: hf_your_token_here
```

### Systemd Service

**/etc/systemd/system/semantic-cache.service:**
```ini
[Unit]
Description=Semantic Cache Service
After=network.target redis.service

[Service]
Type=simple
User=semantic-cache
WorkingDirectory=/opt/semantic-cache
Environment="REDIS_URL=redis://localhost:6379"
Environment="EMBEDDING_MODEL=embeddinggemma"
# For HuggingFace:
# Environment="HF_TOKEN=hf_your_token_here"
# Environment="EMBEDDING_OUTPUT_DIMENSION=768"
ExecStart=/opt/semantic-cache/.venv/bin/uvicorn semantic_cache.api.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Start service:**
```bash
sudo systemctl enable semantic-cache
sudo systemctl start semantic-cache
sudo systemctl status semantic-cache
```

---

## Security Best Practices

### Environment Variables

**DO ✅:**
1. Store secrets in `.env` (add to `.gitignore`)
2. Use read-only HuggingFace tokens
3. Use secret managers in production (AWS Secrets Manager, Vault)
4. Rotate tokens periodically (every 6-12 months)
5. Use separate tokens per environment (dev/staging/prod)

**DON'T ❌:**
1. Commit tokens to git
2. Share tokens in chat/email
3. Use personal tokens for production
4. Hardcode tokens in code
5. Give write access unless needed

### HuggingFace Token Security

**Check if token is exposed:**
```bash
git log -p | grep -i "hf_"
```

**If exposed:**
1. Go to https://huggingface.co/settings/tokens
2. Delete the token immediately
3. Create new token
4. Update `.env`
5. Restart services

### Redis Security

**Enable authentication:**
```yaml
# docker-compose.yml
redis:
  image: redis/redis-stack:latest
  command: redis-server --requirepass your_secure_password
  environment:
    - REDIS_PASSWORD=your_secure_password
```

**Update connection:**
```bash
# .env
REDIS_URL=redis://:your_secure_password@localhost:6379
```

### Network Security

**Production checklist:**
- [ ] Use HTTPS/TLS for API
- [ ] Firewall Redis (only internal access)
- [ ] Rate limiting on API endpoints
- [ ] API authentication (JWT, API keys)
- [ ] Monitor for unusual traffic patterns

---

## Performance Optimization

### Caching Strategy

**Threshold Tuning:**
```bash
# Lower = stricter matching (fewer hits, higher precision)
CACHE_DISTANCE_THRESHOLD=0.10

# Medium = balanced (recommended)
CACHE_DISTANCE_THRESHOLD=0.15

# Higher = looser matching (more hits, risk false positives)
CACHE_DISTANCE_THRESHOLD=0.25
```

**Monitor hit rate:**
```bash
curl http://localhost:8000/cache/stats
```

Target: 60-70% hit rate for optimal cost savings.

### TTL Configuration

```bash
# Short TTL for fast-changing data
CACHE_TTL=3600  # 1 hour

# Medium TTL for typical use
CACHE_TTL=604800  # 7 days (default)

# Long TTL for stable data
CACHE_TTL=2592000  # 30 days
```

### Redis Optimization

**redis.conf settings:**
```conf
# Memory limit
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence (for durability)
save 900 1
save 300 10
save 60 10000

# Snapshotting
rdbcompression yes
rdbchecksum yes
```

### Dimension Selection

**Storage vs Quality:**
- **768 dims**: Best quality, 2x storage
- **512 dims**: 95% quality, 1.3x storage
- **256 dims**: 90% quality, 0.67x storage
- **128 dims**: 85% quality, 0.33x storage

**Recommendation:**
- Small scale (<10K entries): Use 768
- Medium scale (10K-100K): Use 512
- Large scale (>100K): Use 256

### Batch Processing

```python
# Process multiple queries in one embedding call
embeddings = provider.encode_batch([
    "Query 1",
    "Query 2",
    "Query 3"
])
```

Benefits: ~50% faster than individual calls.

---

## Troubleshooting

### Dimension Mismatch Error

**Error:**
```
redis.exceptions.ResponseError: Dimension mismatch: expected 384, got 768
```

**Solution:**
```bash
make cache-clear
make dev
```

### Ollama Connection Refused

**Error:**
```
requests.exceptions.ConnectionError: Connection refused
```

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not, start it
ollama serve
```

### HuggingFace 401 Error

**Error:**
```
HTTPError: 401 Client Error: Unauthorized
```

**Solution:**
```bash
# Authenticate
huggingface-cli login

# Accept model license
# Visit: https://huggingface.co/google/embeddinggemma-300m
# Click: "Agree and access repository"
```

### Redis Connection Error

**Error:**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution:**
```bash
# Check Redis is running
docker compose ps

# Start Redis if needed
docker compose up -d redis

# Check Redis health
redis-cli ping
# Should return: PONG
```

### Slow Inference

**Problem:** Embeddings take >200ms

**Solutions:**
1. **Check model loading**: First request always slower (model loading)
2. **Use batch processing**: Process multiple queries together
3. **Switch to lighter model**: Try `all-minilm` (384 dims, faster)
4. **Check CPU load**: Embeddings are CPU-intensive
5. **Consider GPU**: For production scale

### Memory Issues

**Problem:** High memory usage

**Solutions:**
1. **Use smaller dimensions**: 256 or 128 instead of 768
2. **Limit Redis memory**: Set `maxmemory` in redis.conf
3. **Enable eviction**: Use `allkeys-lru` policy
4. **Monitor with**: `redis-cli info memory`

### Index Creation Fails

**Error:**
```
redis.exceptions.ResponseError: Index already exists
```

**Solution:**
```bash
# Force drop and recreate
redis-cli FT.DROPINDEX semantic_cache DD
make dev
```

---

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Redis health
redis-cli ping

# Ollama health (if using)
curl http://localhost:11434/api/version
```

### Metrics to Track

**Cache Performance:**
- Hit rate (target: 60-70%)
- Average lookup time (target: <5ms)
- Total entries
- Storage size

**API Performance:**
- Request latency (p50, p95, p99)
- Error rate
- Requests per second

**Redis Metrics:**
```bash
redis-cli info stats
redis-cli info memory
redis-cli FT.INFO semantic_cache
```

### Logging

**Application logs:**
```bash
# View logs
docker-compose logs -f api

# Or for systemd
journalctl -u semantic-cache -f
```

**Redis logs:**
```bash
docker-compose logs -f redis
```

---

## Backup and Recovery

### Backup Redis Data

```bash
# Manual backup
redis-cli BGSAVE

# Copy RDB file
cp /var/lib/redis/dump.rdb /backup/dump-$(date +%Y%m%d).rdb
```

### Restore Redis Data

```bash
# Stop Redis
docker-compose stop redis

# Replace RDB file
cp /backup/dump-20260205.rdb /var/lib/redis/dump.rdb

# Start Redis
docker-compose start redis
```

### Automated Backups

**Cron job:**
```bash
# /etc/cron.daily/redis-backup
#!/bin/bash
redis-cli BGSAVE
sleep 10
cp /var/lib/redis/dump.rdb /backup/dump-$(date +%Y%m%d).rdb
find /backup -name "dump-*.rdb" -mtime +7 -delete
```

---

## Multi-Environment Setup

Use different index names per environment:

**.env.dev:**
```bash
CACHE_INDEX_NAME=semantic_cache_dev
```

**.env.staging:**
```bash
CACHE_INDEX_NAME=semantic_cache_staging
```

**.env.prod:**
```bash
CACHE_INDEX_NAME=semantic_cache_prod
```

This allows testing different models/configs without affecting production.

---

## Resources

- [Redis Vector Search Docs](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/)
- [Docker Compose Docs](https://docs.docker.com/compose/)
- [Kubernetes Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Ollama Docker](https://hub.docker.com/r/ollama/ollama)
- [FastAPI Production](https://fastapi.tiangolo.com/deployment/)

---

## Next Steps

1. ✅ Understand production considerations
2. Deploy to your environment
3. Set up monitoring
4. Configure backups
5. Tune performance based on metrics
