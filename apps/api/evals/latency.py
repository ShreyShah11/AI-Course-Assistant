"""
Latency & Performance Metrics
==============================
Tracks timing across all pipeline stages with SLO (Service Level Objective)
compliance checks suitable for production monitoring.

SLO Targets (configurable):
  Planner:          p95 < 4 000 ms
  Embedding:        p95 < 1 500 ms
  Dense retrieval:  p95 < 800 ms
  BM25:             p95 < 200 ms
  MMR:              p95 < 100 ms
  RRF Fusion:       p95 < 100 ms
  End-to-end:       p95 < 8 000 ms
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Default SLOs (all in milliseconds, p95)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SLOS: dict[str, float] = {
    "planner_ms":         4_000,
    "embedding_ms":       1_500,
    "dense_retrieve_ms":    800,
    "bm25_ms":              200,
    "mmr_ms":               100,
    "rrf_ms":               100,
    "total_ms":           8_000,
    # Chunking SLOs (per-document)
    "qna_chunk_ms":       3_000,
    "audio_chunk_ms":     1_000,
    "document_chunk_ms":  5_000,
    "image_ocr_ms":      15_000,  # Tesseract + Gemini Vision
    "handwritten_ocr_ms":10_000,
    "youtube_ingest_ms":  5_000,
}


@dataclass
class LatencyRecord:
    """Single timing observation."""
    stage: str
    ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def within_slo(self) -> bool:
        threshold = DEFAULT_SLOS.get(f"{self.stage}_ms", float("inf"))
        return self.ms <= threshold


@dataclass
class LatencyStats:
    stage: str
    n: int = 0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    slo_ms: float = 0.0
    slo_compliance_rate: float = 0.0   # fraction of observations within SLO

    def summary(self) -> dict:
        return {
            "stage":               self.stage,
            "n":                   self.n,
            "mean_ms":             round(self.mean_ms, 1),
            "std_ms":              round(self.std_ms, 1),
            "min_ms":              round(self.min_ms, 1),
            "max_ms":              round(self.max_ms, 1),
            "p50_ms":              round(self.p50_ms, 1),
            "p95_ms":              round(self.p95_ms, 1),
            "p99_ms":              round(self.p99_ms, 1),
            "slo_ms":              self.slo_ms,
            "slo_compliance_rate": round(self.slo_compliance_rate, 4),
            "slo_passed":          self.p95_ms <= self.slo_ms,
        }


@dataclass
class LatencyReport:
    stages: dict[str, LatencyStats] = field(default_factory=dict)

    def add(self, stage: str, ms: float) -> None:
        """Accumulate a single observation."""
        if stage not in self.stages:
            self.stages[stage] = LatencyStats(stage=stage, slo_ms=DEFAULT_SLOS.get(f"{stage}_ms", float("inf")))
        # We store raw observations in _raw dict for final computation
        if not hasattr(self, "_raw"):
            object.__setattr__(self, "_raw", {})
        self._raw.setdefault(stage, []).append(ms)  # type: ignore[attr-defined]

    def compute(self) -> "LatencyReport":
        """Compute stats from accumulated observations."""
        raw = getattr(self, "_raw", {})
        for stage, observations in raw.items():
            sorted_obs = sorted(observations)
            n = len(sorted_obs)
            slo_ms = DEFAULT_SLOS.get(f"{stage}_ms", float("inf"))
            self.stages[stage] = LatencyStats(
                stage=stage,
                n=n,
                mean_ms=statistics.mean(sorted_obs),
                std_ms=statistics.stdev(sorted_obs) if n > 1 else 0.0,
                min_ms=sorted_obs[0],
                max_ms=sorted_obs[-1],
                p50_ms=_percentile(sorted_obs, 50),
                p95_ms=_percentile(sorted_obs, 95),
                p99_ms=_percentile(sorted_obs, 99),
                slo_ms=slo_ms,
                slo_compliance_rate=sum(1 for v in observations if v <= slo_ms) / n,
            )
        return self

    def summary(self) -> dict:
        return {stage: stats.summary() for stage, stats in self.stages.items()}

    def all_slos_passed(self) -> bool:
        return all(
            stats.p95_ms <= stats.slo_ms
            for stats in self.stages.values()
            if stats.slo_ms < float("inf")
        )


def _percentile(sorted_data: list[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = (len(sorted_data) - 1) * pct / 100
    lower = int(idx)
    upper = lower + 1
    if upper >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[lower] * (1 - (idx - lower)) + sorted_data[upper] * (idx - lower)


class LatencyMetrics:
    """
    Context-manager timer + batch-stats computer.

    Usage (in tests)
    ----------------
    report = LatencyReport()

    with LatencyMetrics.timer("planner") as t:
        plan = run_planner(...)
    report.add("planner", t.elapsed_ms)

    with LatencyMetrics.timer("total") as t:
        result = retrieve(...)
    report.add("total", t.elapsed_ms)

    report.compute()
    assert report.stages["total"].p95_ms <= DEFAULT_SLOS["total_ms"]
    """

    @staticmethod
    @contextmanager
    def timer(stage: str):
        """Context manager that records elapsed time in ms."""
        tracker = _ElapsedTracker(stage)
        t0 = time.perf_counter()
        try:
            yield tracker
        finally:
            tracker.elapsed_ms = (time.perf_counter() - t0) * 1000

    @staticmethod
    def from_retrieval_result(result) -> dict[str, float]:
        """
        Extract per-stage timings from a RetrievalResult if latency_ms is present.
        The retrieval pipeline only exposes total latency; planner ratio is inferred.
        """
        total = getattr(result, "latency_ms", 0.0)
        plan  = getattr(result, "plan", None)
        # Planner ratio is stored in retrieval metrics; approximate here
        return {
            "total_ms": total,
        }

    @staticmethod
    def compute_batch_stats(
        observations: list[float],
        stage: str,
    ) -> LatencyStats:
        """Compute stats from a flat list of ms observations for one stage."""
        if not observations:
            return LatencyStats(stage=stage)
        sorted_obs = sorted(observations)
        n = len(sorted_obs)
        slo_ms = DEFAULT_SLOS.get(f"{stage}_ms", float("inf"))
        return LatencyStats(
            stage=stage,
            n=n,
            mean_ms=statistics.mean(sorted_obs),
            std_ms=statistics.stdev(sorted_obs) if n > 1 else 0.0,
            min_ms=sorted_obs[0],
            max_ms=sorted_obs[-1],
            p50_ms=_percentile(sorted_obs, 50),
            p95_ms=_percentile(sorted_obs, 95),
            p99_ms=_percentile(sorted_obs, 99),
            slo_ms=slo_ms,
            slo_compliance_rate=sum(1 for v in observations if v <= slo_ms) / n,
        )


class _ElapsedTracker:
    def __init__(self, stage: str):
        self.stage = stage
        self.elapsed_ms: float = 0.0