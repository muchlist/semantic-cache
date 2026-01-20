#!/usr/bin/env python3
"""
Demo script for semantic cache.

This script demonstrates the semantic cache functionality with sample queries
in Indonesian and English.
"""

import time

from semantic_cache import EmbeddingService, SemanticCache
from semantic_cache.evaluator import CacheEvaluator, QueryPair


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_cache() -> None:
    """Demonstrate basic cache operations."""
    print_section("Basic Cache Operations")

    cache = SemanticCache()

    # Store some sample Q&A pairs (Indonesian)
    qa_pairs = [
        (
            "Apa itu semantic cache?",
            "Semantic cache adalah sistem caching yang menggunakan kemiripan makna (semantik) untuk menemukan kembali hasil query yang mirip.",
        ),
        (
            "Bagaimana cara kerja embedding?",
            "Embedding mengubah teks menjadi vektor numerik yang merepresentasikan makna, di mana teks dengan makna mirip akan memiliki vektor yang dekat.",
        ),
        (
            "Apa keuntungan menggunakan semantic cache?",
            "Semantic cache dapat mengurangi biaya API LLM hingga 70% dan meningkatkan kecepatan respon dengan menyimpan query yang mirip secara semantik.",
        ),
        (
            "Bagaimana cara menghitung cosine similarity?",
            "Cosine similarity dihitung dengan 1 - cosine_distance, di mana cosine_distance mengukur sudut antara dua vektor.",
        ),
    ]

    print("\n📝 Storing sample Q&A pairs...")
    for prompt, response in qa_pairs:
        cache.store(prompt, response)
        print(f"  ✓ Stored: {prompt[:50]}...")

    # Test cache hits (similar queries in Indonesian)
    print("\n🔍 Testing cache hits (similar queries):")
    test_queries = [
        "Jelaskan apa itu semantic cache",
        "Mengapa semantic cache berguna?",
        "Bagaimana cara menghitung cosine similarity?",
    ]

    for query in test_queries:
        result = cache.check(query)
        match = result.best_match
        if match is not None:
            print(f"\n  Query: {query}")
            print("  ✓ CACHE HIT")
            print(f"  Distance: {match.vector_distance:.4f}")
            print(f"  Similarity: {match.cosine_similarity:.2%}")
            print(f"  Response: {match.response[:100]}...")
        else:
            print(f"\n  Query: {query}")
            print("  ✗ Cache miss")

    # Test cache misses (different queries)
    print("\n🔍 Testing cache misses (different queries):")
    miss_queries = [
        "Apa itu machine learning?",
        "Bagaimana cara deploy aplikasi ke production?",
    ]

    for query in miss_queries:
        result = cache.check(query)
        if result.is_hit:
            print(f"\n  Query: {query}")
            print("  ✗ Unexpected cache hit!")
        else:
            print(f"\n  Query: {query}")
            print("  ✓ Cache miss (expected)")


def demo_embedding_service() -> None:
    """Demonstrate embedding service."""
    print_section("Embedding Service")

    embedding_service = EmbeddingService()

    print("\n📊 Model Information:")
    print(f"  Model: {embedding_service._model_name}")
    print(f"  Dimension: {embedding_service.dimension}")

    # Test with multilingual queries
    queries = [
        "Apa itu semantic cache?",
        "What is semantic cache?",
        "Jelaskan semantic cache",
    ]

    print("\n🔢 Encoding multilingual queries:")
    for query in queries:
        start = time.time()
        embedding = embedding_service.encode(query)
        duration = (time.time() - start) * 1000
        print(f"  '{query}'")
        print(f"    Dimension: {len(embedding)}, Time: {duration:.2f}ms")

    # Compare similarities
    print("\n🔗 Comparing similarities:")
    q1_embed = embedding_service.encode("Apa itu semantic cache?")
    q2_embed = embedding_service.encode("Jelaskan semantic cache")
    q3_embed = embedding_service.encode("What is machine learning?")

    import numpy as np

    sim_12 = 1 - np.dot(q1_embed, q2_embed) / (np.linalg.norm(q1_embed) * np.linalg.norm(q2_embed))
    sim_12 = 1 - sim_12  # Convert back to similarity
    # Actually, for normalized embeddings, cosine similarity = dot product
    sim_12 = float(np.dot(q1_embed, q2_embed))
    sim_13 = float(np.dot(q1_embed, q3_embed))

    print("  'Apa itu semantic cache?' vs 'Jelaskan semantic cache'")
    print(f"    Similarity: {sim_12:.4f}")
    print("  'Apa itu semantic cache?' vs 'What is machine learning?'")
    print(f"    Similarity: {sim_13:.4f}")


def demo_threshold_tuning() -> None:
    """Demonstrate threshold tuning."""
    print_section("Threshold Tuning")

    cache = SemanticCache()

    # Create test query pairs
    test_queries = [
        # Should match (similar meaning)
        QueryPair("Apa itu semantic cache?", "Jelaskan semantic cache", should_match=True),
        QueryPair("Bagaimana embedding bekerja?", "Cara kerja embedding", should_match=True),
        QueryPair("Keuntungan semantic cache?", "Manfaat semantic cache", should_match=True),
        # Should not match (different meaning)
        QueryPair("Apa itu semantic cache?", "Apa itu machine learning?", should_match=False),
        QueryPair("Bagaimana cara deploy?", "Apa itu database?", should_match=False),
    ]

    print(f"\n📋 Test query pairs: {len(test_queries)}")

    # Test different thresholds
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]

    print("\n🎯 Testing thresholds:")
    print(f"{'Threshold':<12} {'Hit Rate':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)

    for threshold in thresholds:
        evaluator = CacheEvaluator(cache)
        result = evaluator.evaluate_threshold(threshold, test_queries)
        print(
            f"{threshold:<12.2f} "
            f"{result.hit_rate:<12.2%} "
            f"{result.precision:<12.2%} "
            f"{result.recall:<12.2%}"
        )
        cache.clear()


def demo_multilingual() -> None:
    """Demonstrate multilingual support."""
    print_section("Multilingual Support (Indonesian/English)")

    cache = SemanticCache()

    # Store Indonesian Q&A
    cache.store(
        "Apa itu semantic cache?",
        "Semantic cache adalah sistem caching pintar yang menggunakan kemiripan makna teks.",
    )

    # Test with English query (should still find semantic match)
    print("\n🌍 Testing cross-language queries:")

    queries = [
        "Apa itu semantic cache?",  # Indonesian (exact)
        "Jelaskan semantic cache",  # Indonesian (similar)
        "What is semantic cache?",  # English (translation)
        "Explain semantic cache",  # English (similar)
    ]

    for query in queries:
        result = cache.check(query)
        match = result.best_match
        if match is not None:
            print(f"\n  Query: '{query}'")
            print(f"  ✓ HIT - Similarity: {match.cosine_similarity:.2%}")
        else:
            print(f"\n  Query: '{query}'")
            print("  ✗ MISS")


def main() -> None:
    """Run all demos."""
    print("\n🚀 Semantic Cache Demo")
    print("=" * 70)
    print("This demo showcases semantic caching with multilingual support")
    print("(Indonesian & English)")

    try:
        demo_basic_cache()
        demo_embedding_service()
        demo_threshold_tuning()
        demo_multilingual()

        print("\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Redis is running:")
        print("  docker compose up -d")
        print("\nOr set REDIS_URL to your Redis instance.")


if __name__ == "__main__":
    main()
