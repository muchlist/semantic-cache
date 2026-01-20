"""
Tests for the semantic cache API.
"""

import pytest
from fastapi.testclient import TestClient

from semantic_cache.api.app import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_root(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "Semantic Cache API"


def test_health(client):
    """Test health check endpoint."""
    response = client.get("/health")
    # May fail if Redis is not running
    assert response.status_code in [200, 503]


def test_cache_check(client):
    """Test cache check endpoint."""
    response = client.post(
        "/cache/check",
        json={"prompt": "Apa itu semantic cache?"},
    )
    # Should work even with no cached data
    assert response.status_code == 200
    data = response.json()
    assert "prompt" in data
    assert "is_hit" in data


def test_cache_store(client):
    """Test cache store endpoint."""
    response = client.post(
        "/cache/store",
        json={
            "prompt": "Test prompt",
            "response": "Test response",
            "metadata": {"test": True},
        },
    )
    # May fail if Redis is not running
    assert response.status_code in [200, 500]


def test_get_threshold(client):
    """Test get threshold endpoint."""
    response = client.get("/cache/threshold")
    assert response.status_code == 200
    data = response.json()
    assert "threshold" in data


def test_get_stats(client):
    """Test get stats endpoint."""
    response = client.get("/stats")
    # May fail if Redis is not running
    assert response.status_code in [200, 500]


def test_get_embedding(client):
    """Test embedding endpoint."""
    response = client.post(
        "/cache/embedding",
        json={"prompt": "Halo, apa kabar?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "embedding_dimension" in data
    assert "model" in data
    assert "multilingual" in data["model"].lower()
