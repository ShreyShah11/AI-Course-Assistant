"""
Retrieval Quality Metrics
=========================
Covers the agentic_retrieval.py pipeline output.

Metrics implemented:
  - MRR@K            (Mean Reciprocal Rank)
  - NDCG@K           (Normalised Discounted Cumulative Gain)
  - Precision@K
  - Recall@K
  - Hit Rate@K       (was ANY relevant chunk retrieved?)
  - Source Coverage  (fraction of expected source namespaces represented)
  - Source Diversity (Shannon entropy across source_key distribution)
  - RRF Score Gap    (score margin between rank-1 and rank-K — proxy for confidence)
  - Sub-query Dedup Rate (fraction of sub-queries that survived deduplication)
  - Planner Latency Ratio (planner ms / total ms)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalMetricResult:
    mrr: float = 0.0
    ndcg: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    hit_rate: float = 0.0
    source_coverage: float = 0.0
    source_diversity: float = 0.0
    rrf_score_gap: float = 0.0
    sub_query_dedup_rate: float = 0.0
    planner_latency_ratio: float = 0.0
    # raw helpers
    total_relevant_retrieved: int = 0
    total_candidates: int = 0
    details: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "MRR":                   round(self.mrr, 4),
            "NDCG@K":                round(self.ndcg, 4),
            "Precision@K":           round(self.precision_at_k, 4),
            "Recall@K":              round(self.recall_at_k, 4),
            "Hit Rate":              round(self.hit_rate, 4),
            "Source Coverage":       round(self.source_coverage, 4),
            "Source Diversity":      round(self.source_diversity, 4),
            "RRF Score Gap":         round(self.rrf_score_gap, 4),
            "SubQuery Dedup Rate":   round(self.sub_query_dedup_rate, 4),
            "Planner Latency Ratio": round(self.planner_latency_ratio, 4),
        }


class RetrievalMetrics:
    """
    Compute retrieval quality metrics given a list of retrieved chunk dicts
    (as returned by RetrievalResult.chunks via .to_dict()) and a ground-truth
    relevance set.

    Ground truth format
    -------------------
    relevant_chunk_ids : set[str]
        chunk_ids that are considered relevant for this query.
        For tests without a live index these are provided in the fixture.

    relevance_grades : dict[str, int]   (optional, for NDCG)
        chunk_id → grade (0=irrelevant, 1=partial, 2=highly_relevant).
        If not supplied, binary relevance is assumed (grade=2 for relevant, 0 otherwise).
    """

    @staticmethod
    def mrr(retrieved: list[dict], relevant_ids: set[str]) -> float:
        """Mean Reciprocal Rank — rewards the rank of the first relevant hit."""
        for rank, chunk in enumerate(retrieved, start=1):
            if chunk.get("chunk_id") in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg(
        retrieved: list[dict],
        relevance_grades: dict[str, int],
        k: Optional[int] = None,
    ) -> float:
        """NDCG@K — graded relevance weighted by log-rank discount."""
        k = k or len(retrieved)
        retrieved_k = retrieved[:k]

        def dcg(items: list[dict]) -> float:
            score = 0.0
            for i, chunk in enumerate(items, start=1):
                rel = relevance_grades.get(chunk.get("chunk_id", ""), 0)
                score += (2**rel - 1) / math.log2(i + 1)
            return score

        actual_dcg = dcg(retrieved_k)
        ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
        ideal_items = [{"chunk_id": f"__ideal_{i}"} for i in range(len(ideal_grades))]
        ideal_rel_map = {f"__ideal_{i}": g for i, g in enumerate(ideal_grades)}

        # build ideal list
        ideal_dcg_val = 0.0
        for i, (item, grade) in enumerate(zip(ideal_items, ideal_grades), start=1):
            ideal_dcg_val += (2**grade - 1) / math.log2(i + 1)

        if ideal_dcg_val == 0:
            return 0.0
        return actual_dcg / ideal_dcg_val

    @staticmethod
    def precision_at_k(retrieved: list[dict], relevant_ids: set[str], k: int) -> float:
        """Fraction of top-K retrieved chunks that are relevant."""
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for c in top_k if c.get("chunk_id") in relevant_ids)
        return hits / len(top_k)

    @staticmethod
    def recall_at_k(
        retrieved: list[dict], relevant_ids: set[str], k: int
    ) -> float:
        """Fraction of all relevant chunks found in top-K."""
        if not relevant_ids:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for c in top_k if c.get("chunk_id") in relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def hit_rate(retrieved: list[dict], relevant_ids: set[str]) -> float:
        """Binary: 1.0 if at least one relevant chunk was retrieved, else 0.0."""
        for chunk in retrieved:
            if chunk.get("chunk_id") in relevant_ids:
                return 1.0
        return 0.0

    @staticmethod
    def source_coverage(
        retrieved: list[dict], expected_sources: set[str]
    ) -> float:
        """Fraction of expected source namespaces that appear in results."""
        if not expected_sources:
            return 1.0
        found = {c.get("source_key", "") for c in retrieved}
        return len(found & expected_sources) / len(expected_sources)

    @staticmethod
    def source_diversity(retrieved: list[dict]) -> float:
        """
        Shannon entropy of source_key distribution (normalised to [0,1]).
        Higher = more diverse across source types.
        """
        from collections import Counter
        counts = Counter(c.get("source_key", "unknown") for c in retrieved)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = -sum((v / total) * math.log2(v / total) for v in counts.values())
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def rrf_score_gap(retrieved: list[dict]) -> float:
        """
        Score margin between rank-1 and rank-K chunk.
        Large gap → model is confident about the top result.
        """
        scores = [c.get("score", 0.0) for c in retrieved if "score" in c]
        if len(scores) < 2:
            return 0.0
        return scores[0] - scores[-1]

    @staticmethod
    def sub_query_dedup_rate(
        original_count: int, deduplicated_count: int
    ) -> float:
        """
        Fraction of sub-queries that survived deduplication.
        Near 1.0 → planner generated diverse sub-queries (good).
        """
        if original_count == 0:
            return 0.0
        return deduplicated_count / original_count

    @staticmethod
    def planner_latency_ratio(planner_ms: float, total_ms: float) -> float:
        """
        Fraction of total latency consumed by the planner.
        Healthy: < 0.40 (planner shouldn't dominate).
        """
        if total_ms <= 0:
            return 0.0
        return min(planner_ms / total_ms, 1.0)

    @classmethod
    def evaluate(
        cls,
        retrieved: list[dict],
        relevant_ids: set[str],
        expected_sources: set[str],
        k: int = 8,
        relevance_grades: Optional[dict[str, int]] = None,
        original_sub_query_count: int = 0,
        deduplicated_sub_query_count: int = 0,
        planner_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> RetrievalMetricResult:
        """
        Compute all retrieval metrics in one call.

        Parameters
        ----------
        retrieved            : list of chunk dicts (.to_dict() output)
        relevant_ids         : ground-truth relevant chunk IDs
        expected_sources     : source_key values that should appear in results
        k                    : cutoff for @K metrics
        relevance_grades     : chunk_id → int grade (optional; falls back to binary)
        original_sub_query_count    : sub-queries before dedup
        deduplicated_sub_query_count: sub-queries after dedup
        planner_ms / total_ms : latency breakdown from RetrievalResult
        """
        grades = relevance_grades or {cid: 2 for cid in relevant_ids}

        result = RetrievalMetricResult(
            mrr=cls.mrr(retrieved, relevant_ids),
            ndcg=cls.ndcg(retrieved, grades, k),
            precision_at_k=cls.precision_at_k(retrieved, relevant_ids, k),
            recall_at_k=cls.recall_at_k(retrieved, relevant_ids, k),
            hit_rate=cls.hit_rate(retrieved, relevant_ids),
            source_coverage=cls.source_coverage(retrieved, expected_sources),
            source_diversity=cls.source_diversity(retrieved),
            rrf_score_gap=cls.rrf_score_gap(retrieved),
            sub_query_dedup_rate=cls.sub_query_dedup_rate(
                original_sub_query_count, deduplicated_sub_query_count
            ),
            planner_latency_ratio=cls.planner_latency_ratio(planner_ms, total_ms),
            total_relevant_retrieved=sum(
                1 for c in retrieved[:k] if c.get("chunk_id") in relevant_ids
            ),
            total_candidates=len(retrieved),
        )
        return result