.PHONY: help install dev demo test lint format type-check clean redis-up redis-down redis-logs deps

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := uv run python
API_HOST := 0.0.0.0
API_PORT := 8000

##@ General
help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development
install: ## Install dependencies
	@echo "Installing dependencies..."
	uv sync

dev: ## Run development server (FastAPI with hot reload)
	@echo "Starting development server on http://$(API_HOST):$(API_PORT)"
	uv run uvicorn semantic_cache.api.app:app --host $(API_HOST) --port $(API_PORT) --reload

demo: ## Run demo script
	@echo "Running semantic cache demo..."
	uv run python scripts/demo.py

##@ Redis
redis-up: ## Start Redis with docker compose
	@echo "Starting Redis..."
	docker-compose up -d redis

redis-down: ## Stop Redis
	@echo "Stopping Redis..."
	docker-compose down

redis-logs: ## Show Redis logs
	docker-compose logs -f redis

redis-restart: ## Restart Redis
	@echo "Restarting Redis..."
	docker-compose restart redis

##@ Testing
test: ## Run tests
	@echo "Running tests..."
	uv run pytest -v

test-cov: ## Run tests with coverage
	@echo "Running tests with coverage..."
	uv run pytest --cov=semantic_cache --cov-report=html --cov-report=term

##@ Code Quality
lint: ## Run linter (ruff)
	@echo "Linting code..."
	uv run ruff check src/ tests/ scripts/

format: ## Format code with ruff
	@echo "Formatting code..."
	uv run ruff format src/ tests/ scripts/

format-check: ## Check if code is formatted
	@echo "Checking code formatting..."
	uv run ruff format --check src/ tests/ scripts/

type-check: ## Run type checker (ty)
	@echo "Running type checker..."
	uv run ty check src/

check: lint type-check ## Run all checks (lint + type-check)

fix: ## Auto-fix linting issues
	@echo "Auto-fixing linting issues..."
	uv run ruff check --fix src/ tests/ scripts/

##@ Database
cache-clear: ## Clear all cache entries
	@echo "Clearing cache..."
	@curl -X DELETE http://localhost:$(API_PORT)/cache 2>/dev/null || echo "API not running. Start with 'make dev'"

cache-stats: ## Show cache statistics
	@echo "Fetching cache stats..."
	@curl -s http://localhost:$(API_PORT)/stats | python3 -m json.tool 2>/dev/null || curl -s http://localhost:$(API_PORT)/stats

cache-health: ## Check cache health
	@echo "Checking cache health..."
	@curl -s http://localhost:$(API_PORT)/health

##@ API Testing
api-check: ## Test cache check endpoint
	@echo "Testing cache check..."
	@curl -X POST http://localhost:$(API_PORT)/cache/check \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Apa itu semantic cache?"}' | python3 -m json.tool

api-store: ## Test cache store endpoint
	@echo "Testing cache store..."
	@curl -X POST http://localhost:$(API_PORT)/cache/store \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Test prompt", "response": "Test response"}' | python3 -m json.tool

api-docs: ## Open API documentation (in browser)
	@echo "Opening API docs at http://localhost:$(API_PORT)/docs"
	@open http://localhost:$(API_PORT)/docs 2>/dev/null || echo "Open http://localhost:$(API_PORT)/docs in your browser"

##@ Utilities
deps: ## Update dependencies
	@echo "Updating dependencies..."
	uv sync --upgrade

clean: ## Clean up cache and build artifacts
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "Clean complete"

clean-all: clean ## Clean everything including Redis data
	@echo "Cleaning everything..."
	docker-compose down -v
	rm -rf .venv 2>/dev/null || true
	@echo "Deep clean complete"

##@ Setup
setup: install redis-up ## Initial project setup
	@echo ""
	@echo "Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make dev' to start the API server"
	@echo "  2. Run 'make demo' to see the demo"
	@echo "  3. Open http://localhost:8000/docs for API documentation"
	@echo ""
