"""
Evals package — evaluation metrics for the CourseGPT RAG pipeline.

Quick imports
-------------
from evals import ChunkingMetrics, RetrievalMetrics, FaithfulnessMetrics
from evals import CoverageMetrics, LatencyMetrics
from evals.runner import run_eval_suite
"""

from evals.chunking_metrics import (
    ChunkingMetrics,
    ChunkingMetricResult,
    ChunkSizeStats,
    MetadataCompletenessResult,
)
from evals.retrieval_metrics import RetrievalMetrics, RetrievalMetricResult
from evals.faithfulness import FaithfulnessMetrics, FaithfulnessResult
from evals.coverage import CoverageMetrics, CoverageResult
from evals.latency import (
    LatencyMetrics,
    LatencyReport,
    LatencyStats,
    LatencyRecord,
    DEFAULT_SLOS,
)

__all__ = [
    # Chunking
    "ChunkingMetrics",
    "ChunkingMetricResult",
    "ChunkSizeStats",
    "MetadataCompletenessResult",
    # Retrieval
    "RetrievalMetrics",
    "RetrievalMetricResult",
    # Faithfulness
    "FaithfulnessMetrics",
    "FaithfulnessResult",
    # Coverage
    "CoverageMetrics",
    "CoverageResult",
    # Latency
    "LatencyMetrics",
    "LatencyReport",
    "LatencyStats",
    "LatencyRecord",
    "DEFAULT_SLOS",
]
