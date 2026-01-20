"""
Evaluation utilities for semantic cache.

This module provides tools for evaluating cache performance, tuning thresholds,
and measuring the effectiveness of semantic caching.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_cache.cache import SemanticCache
from semantic_cache.models import PerformanceMetrics


@dataclass
class EvalResult:
    """Result of a cache evaluation run."""

    threshold: float
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    avg_lookup_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def precision(self) -> float:
        """Calculate precision (TP / (TP + FP))."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Calculate recall (TP / (TP + FN))."""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def f1_score(self) -> float:
        """Calculate F1 score (2 * precision * recall / (precision + recall))."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "threshold": self.threshold,
            "hit_rate": self.hit_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "avg_lookup_time_ms": self.avg_lookup_time_ms,
        }


@dataclass
class QueryPair:
    """A pair of queries with their expected match relationship."""

    query: str
    cached_query: str
    should_match: bool  # True if semantically similar, False if different


class CacheEvaluator:
    """Evaluator for semantic cache performance."""

    def __init__(
        self,
        cache: SemanticCache,
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            cache: The SemanticCache instance to evaluate.
        """
        self.cache = cache
        self.results: list[EvalResult] = []

    def evaluate_threshold(
        self,
        threshold: float,
        test_queries: list[QueryPair],
    ) -> EvalResult:
        """
        Evaluate cache performance at a specific threshold.

        Args:
            threshold: The distance threshold to test.
            test_queries: List of QueryPair objects to test.

        Returns:
            EvalResult with metrics for this threshold.
        """
        result = EvalResult(threshold=threshold)
        total_lookup_time = 0.0

        # Store the cached queries first
        for pair in test_queries:
            self.cache.store(
                pair.cached_query,
                f"Response for: {pair.cached_query}",
            )

        # Test each query
        for pair in test_queries:
            start_time = time.time()
            cache_result = self.cache.check(pair.query, distance_threshold=threshold)
            lookup_time_ms = (time.time() - start_time) * 1000
            total_lookup_time += lookup_time_ms

            result.total_queries += 1

            if cache_result.is_hit:
                result.cache_hits += 1
                # Check if this is a true positive (should match and did)
                if pair.should_match:
                    result.true_positives += 1
                else:
                    result.false_positives += 1
            else:
                result.cache_misses += 1
                # Check if this is a false negative (should match but didn't)
                if pair.should_match:
                    result.false_negatives += 1
                else:
                    result.true_negatives += 1

        if result.total_queries > 0:
            result.avg_lookup_time_ms = total_lookup_time / result.total_queries

        self.results.append(result)
        return result

    def sweep_thresholds(
        self,
        test_queries: list[QueryPair],
        min_threshold: float = 0.05,
        max_threshold: float = 0.50,
        steps: int = 10,
    ) -> list[EvalResult]:
        """
        Sweep across multiple threshold values to find optimal.

        Args:
            test_queries: List of QueryPair objects to test.
            min_threshold: Minimum threshold to test.
            max_threshold: Maximum threshold to test.
            steps: Number of threshold steps to test.

        Returns:
            List of EvalResult for each threshold tested.
        """
        # Clear previous results
        self.results = []
        self.cache.clear()

        thresholds = np.linspace(min_threshold, max_threshold, steps)

        for threshold in thresholds:
            print(f"Testing threshold: {threshold:.3f}")
            result = self.evaluate_threshold(threshold, test_queries)
            print(
                f"  Hit Rate: {result.hit_rate:.2%}, "
                f"Precision: {result.precision:.2%}, "
                f"F1: {result.f1_score:.2%}"
            )
            self.cache.clear()

        return self.results

    def find_optimal_threshold(
        self,
        metric: str = "f1_score",
    ) -> tuple[float, EvalResult]:
        """
        Find the optimal threshold based on a metric.

        Args:
            metric: Metric to optimize ('f1_score', 'precision', 'recall', 'hit_rate').

        Returns:
            Tuple of (threshold, result) for the optimal threshold.
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run sweep_thresholds first.")

        best_result = max(self.results, key=lambda r: getattr(r, metric))
        return best_result.threshold, best_result

    def print_summary(self) -> None:
        """Print a summary of all evaluation results."""
        if not self.results:
            print("No evaluation results available.")
            return

        print("\n" + "=" * 80)
        print("Cache Evaluation Summary")
        print("=" * 80)
        print(
            f"{'Threshold':<12} {'Hit Rate':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}"
        )
        print("-" * 80)

        for result in self.results:
            print(
                f"{result.threshold:<12.3f} "
                f"{result.hit_rate:<12.2%} "
                f"{result.precision:<12.2%} "
                f"{result.recall:<12.2%} "
                f"{result.f1_score:<12.2%}"
            )

        print("=" * 80)

        # Find best by different metrics
        for metric in ["f1_score", "precision", "recall", "hit_rate"]:
            threshold, result = self.find_optimal_threshold(metric)
            print(f"Best {metric}: {threshold:.3f} ({result.__getattribute__(metric):.2%})")


class SimplePerfEval:
    """Simple performance evaluator for cache operations."""

    def __init__(self) -> None:
        """Initialize the performance evaluator."""
        self.metrics = PerformanceMetrics()
        self.llm_records: list[dict[str, Any]] = []
        self._start_time: float | None = None
        self._operation_start: float | None = None

    def start(self) -> None:
        """Start timing an operation."""
        self._operation_start = time.time()

    def tick(self, label: str) -> float:
        """
        Record an operation and return its duration.

        Args:
            label: The label for this operation (e.g., 'cache_hit', 'llm_call').

        Returns:
            Duration in milliseconds.
        """
        if self._operation_start is None:
            raise RuntimeError("Must call start() before tick()")

        duration_ms = (time.time() - self._operation_start) * 1000
        self._operation_start = None

        if label == "cache_hit":
            self.metrics.record_hit(duration_ms)
        elif label == "cache_miss":
            self.metrics.record_miss(duration_ms)

        return duration_ms

    def record_llm_call(self, model: str, prompt: str, response: str, duration_ms: float) -> None:
        """Record an LLM API call."""
        self.metrics.record_llm_call(duration_ms)
        self.llm_records.append(
            {
                "model": model,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response[:100] + "..." if len(response) > 100 else response,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            }
        )

    def get_metrics(self) -> dict[str, float | int]:
        """Get current performance metrics."""
        return self.metrics.to_dict()

    def get_costs(self) -> dict[str, float]:
        """
        Estimate LLM API costs.

        Note: This is a simplified cost estimation based on typical GPT pricing.
        Adjust for your actual provider and model.
        """
        # Simplified cost estimation (adjust for your actual model/pricing)
        input_cost_per_1k = 0.00015  # ~$0.15 per 1M input tokens (GPT-4o-mini)
        output_cost_per_1k = 0.00060  # ~$0.60 per 1M output tokens

        # Rough estimation: 1 token ≈ 4 characters
        total_input_tokens = sum(len(r["prompt"]) // 4 for r in self.llm_records)
        total_output_tokens = sum(len(r["response"]) // 4 for r in self.llm_records)

        input_cost = (total_input_tokens / 1000) * input_cost_per_1k
        output_cost = (total_output_tokens / 1000) * output_cost_per_1k

        return {
            "total_llm_calls": self.metrics.llm_calls,
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "estimated_cost_usd": input_cost + output_cost,
        }

    def summary(self) -> str:
        """Get a formatted summary of metrics."""
        m = self.metrics
        costs = self.get_costs()

        return f"""
Performance Summary:
====================
Total Queries:        {m.total_queries}
Cache Hits:           {m.cache_hits} ({m.hit_rate:.1%})
Cache Misses:         {m.cache_misses}
Avg Lookup Time:      {m.avg_lookup_time_ms:.2f}ms
LLM Calls:            {m.llm_calls}
Total LLM Time:       {m.total_llm_time_ms:.0f}ms
Estimated Cost:       ${costs["estimated_cost_usd"]:.4f}
"""
