"""
Chunking Quality Metrics
========================
Evaluates the output of ALL six ingestion chunking pipelines:
  - QnA pipeline (ingestion_pipeline.py / QnAChunk)
  - Audio pipeline (ingestion_pipeline.py / CourseChunk)
  - Document pipeline (ingestion_pipeline_pinecone.py / LangChain Document)
  - Image pipeline (image_ingestion_pipeline.py / Chunk)
  - Handwritten pipeline (handwritten_pipeline.py / Chunk)
  - YouTube pipeline (ingestion_pipeline.py / TranscriptChunk)

Metrics
-------
  ChunkSizeStats      : token distribution (mean, std, p5, p95, over/under rate)
  OverlapIntegrity    : for sliding-window chunks — shared token ratio
  MetadataCompleteness: fraction of required metadata fields that are non-null
  ChunkTypeBalance    : distribution of chunk_type values (QnA only)
  QnAExtractionRate   : fraction of blocks that resolved to Q/A pairs (vs passage)
  SemanticClusterQuality: intra-cluster vs inter-cluster cosine similarity (QnA)
  SlidingWindowCoverage : every transcript second covered by ≥1 chunk (Audio)
  DeduplicationRate   : SHA-256 ID collisions / total chunks (Document)
  OCRConfidenceStats  : confidence percentile stats (Image / Handwritten)
  ChapterAlignmentRate: fraction of YouTube chunks that match a chapter boundary
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkSizeStats:
    mean_tokens: float = 0.0
    std_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    p5_tokens: float = 0.0
    p95_tokens: float = 0.0
    over_limit_rate: float = 0.0   # fraction above max_limit
    under_limit_rate: float = 0.0  # fraction below min_limit
    total_chunks: int = 0

    def summary(self) -> dict:
        return {
            "mean_tokens":     round(self.mean_tokens, 1),
            "std_tokens":      round(self.std_tokens, 1),
            "min_tokens":      self.min_tokens,
            "max_tokens":      self.max_tokens,
            "p5_tokens":       round(self.p5_tokens, 1),
            "p95_tokens":      round(self.p95_tokens, 1),
            "over_limit_rate": round(self.over_limit_rate, 4),
            "under_limit_rate": round(self.under_limit_rate, 4),
            "total_chunks":    self.total_chunks,
        }


@dataclass
class MetadataCompletenessResult:
    field_presence: dict[str, float] = field(default_factory=dict)  # field → presence rate
    overall_completeness: float = 0.0
    missing_critical_fields: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "overall_completeness":   round(self.overall_completeness, 4),
            "missing_critical_fields": self.missing_critical_fields,
            "field_presence":          {k: round(v, 4) for k, v in self.field_presence.items()},
        }


@dataclass
class ChunkingMetricResult:
    pipeline: str = ""
    chunk_size_stats: ChunkSizeStats = field(default_factory=ChunkSizeStats)
    metadata_completeness: MetadataCompletenessResult = field(
        default_factory=MetadataCompletenessResult
    )
    # Pipeline-specific
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "pipeline":             self.pipeline,
            "chunk_size_stats":     self.chunk_size_stats.summary(),
            "metadata_completeness": self.metadata_completeness.summary(),
            **self.extras,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word (GPT-style)."""
    return max(1, int(len(text.split()) * 1.3))


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * pct / 100
    lower = int(idx)
    upper = lower + 1
    if upper >= len(sorted_data):
        return sorted_data[-1]
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def _chunk_size_stats(
    token_counts: list[int],
    min_limit: int = 0,
    max_limit: int = 99_999,
) -> ChunkSizeStats:
    if not token_counts:
        return ChunkSizeStats()
    counts_f = [float(c) for c in token_counts]
    mean = statistics.mean(counts_f)
    std  = statistics.stdev(counts_f) if len(counts_f) > 1 else 0.0
    return ChunkSizeStats(
        mean_tokens=mean,
        std_tokens=std,
        min_tokens=min(token_counts),
        max_tokens=max(token_counts),
        p5_tokens=_percentile(counts_f, 5),
        p95_tokens=_percentile(counts_f, 95),
        over_limit_rate=sum(1 for c in token_counts if c > max_limit) / len(token_counts),
        under_limit_rate=sum(1 for c in token_counts if c < min_limit) / len(token_counts),
        total_chunks=len(token_counts),
    )


def _metadata_completeness(
    chunks: list[dict],
    required_fields: list[str],
    critical_fields: list[str],
) -> MetadataCompletenessResult:
    if not chunks:
        return MetadataCompletenessResult()

    field_hits: dict[str, int] = {f: 0 for f in required_fields}
    for chunk in chunks:
        for f in required_fields:
            val = chunk.get(f)
            # non-null, non-empty-string, non-empty-list
            if val is not None and val != "" and val != [] and val != {}:
                field_hits[f] += 1

    n = len(chunks)
    field_presence = {f: field_hits[f] / n for f in required_fields}
    overall = statistics.mean(field_presence.values()) if field_presence else 0.0
    missing_critical = [
        f for f in critical_fields if field_presence.get(f, 0.0) < 0.90
    ]
    return MetadataCompletenessResult(
        field_presence=field_presence,
        overall_completeness=overall,
        missing_critical_fields=missing_critical,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-specific evaluators
# ─────────────────────────────────────────────────────────────────────────────

class ChunkingMetrics:

    # ── QnA Pipeline ─────────────────────────────────────────────────────────

    @staticmethod
    def evaluate_qna(chunks: list[dict], max_tokens: int = 800) -> ChunkingMetricResult:
        """
        chunks: list of dicts from process_document() → asdict(QnAChunk)
        Required keys: chunk_id, source_file, file_type, page_range, chunk_type,
                       topic_cluster, topic_label, questions, answers, raw_text,
                       char_count, token_estimate, difficulty_hint, has_sub_parts,
                       marks_hint, year_hint
        """
        required = [
            "chunk_id", "source_file", "file_type", "chunk_type",
            "topic_cluster", "topic_label", "questions", "raw_text",
            "char_count", "token_estimate",
        ]
        critical = ["chunk_id", "source_file", "chunk_type", "questions", "raw_text"]

        token_counts = [c.get("token_estimate", _approx_tokens(c.get("raw_text", "")))
                        for c in chunks]
        size_stats = _chunk_size_stats(token_counts, min_limit=20, max_limit=max_tokens)
        meta = _metadata_completeness(chunks, required, critical)

        # QnA-specific metrics
        chunk_type_dist: dict[str, int] = {}
        qa_extraction_rate = 0
        has_answers_rate = 0
        for c in chunks:
            ct = c.get("chunk_type", "unknown")
            chunk_type_dist[ct] = chunk_type_dist.get(ct, 0) + 1
            if ct in ("single_qa", "multi_qa"):
                qa_extraction_rate += 1
            if c.get("answers"):
                has_answers_rate += 1

        n = len(chunks) or 1
        extras = {
            "chunk_type_distribution": chunk_type_dist,
            "qa_extraction_rate": round(qa_extraction_rate / n, 4),
            "answer_presence_rate": round(has_answers_rate / n, 4),
            "year_hint_coverage": round(
                sum(1 for c in chunks if c.get("year_hint")) / n, 4
            ),
            "marks_hint_coverage": round(
                sum(1 for c in chunks if c.get("marks_hint")) / n, 4
            ),
            "avg_questions_per_chunk": round(
                statistics.mean([len(c.get("questions", [])) for c in chunks]) if chunks else 0, 2
            ),
        }

        return ChunkingMetricResult(
            pipeline="qna",
            chunk_size_stats=size_stats,
            metadata_completeness=meta,
            extras=extras,
        )

    # ── Audio Pipeline ────────────────────────────────────────────────────────

    @staticmethod
    def evaluate_audio(chunks: list[dict], lecture_duration_sec: float = 0.0) -> ChunkingMetricResult:
        """
        chunks: list of CourseChunk.to_dict() output.
        lecture_duration_sec: total lecture length for coverage check.
        """
        required = [
            "chunk_id", "course_id", "lecture_id", "strategy",
            "text", "word_count", "start_seconds", "end_seconds",
            "segment_indices", "concepts", "avg_confidence",
        ]
        critical = [
            "chunk_id", "lecture_id", "strategy", "text", "avg_confidence",
        ]

        token_counts = [_approx_tokens(c.get("text", "")) for c in chunks]
        size_stats = _chunk_size_stats(token_counts, min_limit=30, max_limit=600)
        meta = _metadata_completeness(chunks, required, critical)

        # Strategy breakdown
        strategy_dist: dict[str, int] = {}
        for c in chunks:
            s = c.get("strategy", "unknown")
            strategy_dist[s] = strategy_dist.get(s, 0) + 1

        # Overlap verification for sliding_window chunks
        sw_chunks = [c for c in chunks if c.get("strategy") == "sliding_window"]
        overlap_ok_rate = 0.0
        if sw_chunks:
            ok = sum(1 for c in sw_chunks if c.get("overlap_with_prev", 0) >= 0)
            overlap_ok_rate = ok / len(sw_chunks)

        # Coverage: fraction of lecture duration covered
        coverage = 0.0
        if lecture_duration_sec > 0 and chunks:
            max_end = max(c.get("end_seconds", 0.0) for c in chunks)
            coverage = min(max_end / lecture_duration_sec, 1.0)

        # Confidence stats
        conf_vals = [c.get("avg_confidence", 0.0) for c in chunks if "avg_confidence" in c]
        avg_conf = statistics.mean(conf_vals) if conf_vals else 0.0
        low_conf_rate = sum(1 for v in conf_vals if v < 0.7) / len(conf_vals) if conf_vals else 0.0

        extras = {
            "strategy_distribution": strategy_dist,
            "sliding_window_overlap_ok_rate": round(overlap_ok_rate, 4),
            "lecture_temporal_coverage": round(coverage, 4),
            "avg_confidence": round(avg_conf, 4),
            "low_confidence_chunk_rate": round(low_conf_rate, 4),
            "concept_coverage_rate": round(
                sum(1 for c in chunks if c.get("concepts")) / (len(chunks) or 1), 4
            ),
        }

        return ChunkingMetricResult(
            pipeline="audio",
            chunk_size_stats=size_stats,
            metadata_completeness=meta,
            extras=extras,
        )

    # ── Document Pipeline ─────────────────────────────────────────────────────

    @staticmethod
    def evaluate_documents(chunks: list[dict]) -> ChunkingMetricResult:
        """
        chunks: list of Document.metadata dicts from build_documents().
        """
        required = [
            "chunk_id", "source", "file_name", "file_type", "chunk_index",
            "total_chunks", "page_start", "page_end", "section_title",
            "char_count", "word_count", "keywords", "raw_text", "ingested_at",
        ]
        critical = [
            "chunk_id", "source", "file_type", "raw_text", "chunk_index",
        ]

        token_counts = [_approx_tokens(c.get("raw_text", "")) for c in chunks]
        size_stats = _chunk_size_stats(token_counts, min_limit=40, max_limit=900)
        meta = _metadata_completeness(chunks, required, critical)

        n = len(chunks) or 1
        extras = {
            "has_table_rate":   round(sum(1 for c in chunks if c.get("has_table")) / n, 4),
            "has_image_rate":   round(sum(1 for c in chunks if c.get("has_image")) / n, 4),
            "has_list_rate":    round(sum(1 for c in chunks if c.get("has_list")) / n, 4),
            "section_title_coverage": round(
                sum(1 for c in chunks if c.get("section_title")) / n, 4
            ),
            "keywords_coverage": round(
                sum(1 for c in chunks if c.get("keywords")) / n, 4
            ),
            # Deterministic ID check: chunk_ids should all be unique
            "unique_chunk_id_rate": round(
                len({c.get("chunk_id") for c in chunks}) / n, 4
            ),
            # Image summaries present when images detected
            "image_summary_coverage": round(
                sum(
                    1 for c in chunks
                    if c.get("has_image") and c.get("image_summaries") not in (None, "[]", "")
                ) / max(sum(1 for c in chunks if c.get("has_image")), 1),
                4
            ),
        }

        return ChunkingMetricResult(
            pipeline="documents",
            chunk_size_stats=size_stats,
            metadata_completeness=meta,
            extras=extras,
        )

    # ── Image Pipeline ────────────────────────────────────────────────────────

    @staticmethod
    def evaluate_image(
        chunks: list[dict],
        ocr_confidence_threshold: float = 80.0,
    ) -> ChunkingMetricResult:
        """
        chunks: list of Chunk.to_dict() output from ChunkingPipeline.run().
        """
        required = [
            "text", "metadata",
        ]
        # Flatten metadata if nested
        flat_chunks: list[dict] = []
        for c in chunks:
            if "metadata" in c and isinstance(c["metadata"], dict):
                flat = {**c["metadata"], "text": c.get("text", "")}
            else:
                flat = c
            flat_chunks.append(flat)

        meta_required = [
            "chunk_id", "source_file", "page_numbers", "char_count",
            "word_count", "ocr_confidence", "image_type",
        ]
        critical = ["chunk_id", "source_file", "ocr_confidence"]
        meta = _metadata_completeness(flat_chunks, meta_required, critical)

        token_counts = [_approx_tokens(c.get("text", "")) for c in flat_chunks]
        size_stats = _chunk_size_stats(token_counts, min_limit=25, max_limit=500)

        conf_vals = [
            float(c.get("ocr_confidence", 0.0)) for c in flat_chunks
            if "ocr_confidence" in c
        ]
        n = len(flat_chunks) or 1
        extras = {
            "avg_ocr_confidence": round(statistics.mean(conf_vals) if conf_vals else 0.0, 2),
            "low_confidence_rate": round(
                sum(1 for v in conf_vals if v < ocr_confidence_threshold) / (len(conf_vals) or 1), 4
            ),
            "has_code_rate":    round(sum(1 for c in flat_chunks if c.get("has_code")) / n, 4),
            "has_formula_rate": round(sum(1 for c in flat_chunks if c.get("has_formula")) / n, 4),
            "prev_next_link_rate": round(
                sum(1 for c in flat_chunks if c.get("prev_chunk_id") or c.get("next_chunk_id")) / n, 4
            ),
            "gemini_enrichment_rate": round(
                sum(1 for c in flat_chunks if c.get("gemini_summary") or c.get("gemini_topic")) / n, 4
            ),
        }

        return ChunkingMetricResult(
            pipeline="image",
            chunk_size_stats=size_stats,
            metadata_completeness=meta,
            extras=extras,
        )

    # ── Handwritten Pipeline ──────────────────────────────────────────────────

    @staticmethod
    def evaluate_handwritten(chunks: list[dict]) -> ChunkingMetricResult:
        """
        chunks: list of Chunk.metadata() dicts from the handwritten pipeline.
        """
        required = [
            "course_id", "source_file", "page_number", "chunk_index",
            "topic", "content_type", "confidence", "model_used",
            "has_diagrams", "has_equations", "has_tables",
            "ink_quality", "writing_style",
        ]
        critical = [
            "course_id", "source_file", "confidence", "model_used",
        ]
        meta = _metadata_completeness(chunks, required, critical)

        token_counts = [_approx_tokens(c.get("text_preview", "")) for c in chunks]
        size_stats = _chunk_size_stats(token_counts, min_limit=20, max_limit=700)

        conf_vals = [float(c.get("confidence", 0.5)) for c in chunks]
        flash_count = sum(
            1 for c in chunks if "flash" in c.get("model_used", "").lower()
        )
        pro_count = sum(
            1 for c in chunks if "pro" in c.get("model_used", "").lower()
        )
        n = len(chunks) or 1

        content_type_dist: dict[str, int] = {}
        for c in chunks:
            ct = c.get("content_type", "unknown")
            content_type_dist[ct] = content_type_dist.get(ct, 0) + 1

        extras = {
            "avg_confidence":    round(statistics.mean(conf_vals) if conf_vals else 0.0, 4),
            "flash_pages_rate":  round(flash_count / n, 4),
            "pro_escalation_rate": round(pro_count / n, 4),
            "has_equations_rate": round(sum(1 for c in chunks if c.get("has_equations")) / n, 4),
            "has_diagrams_rate":  round(sum(1 for c in chunks if c.get("has_diagrams")) / n, 4),
            "content_type_distribution": content_type_dist,
            "ink_quality_good_rate": round(
                sum(1 for c in chunks if c.get("ink_quality") == "good") / n, 4
            ),
        }

        return ChunkingMetricResult(
            pipeline="handwritten",
            chunk_size_stats=size_stats,
            metadata_completeness=meta,
            extras=extras,
        )

    # ── YouTube Pipeline ──────────────────────────────────────────────────────

    @staticmethod
    def evaluate_youtube(
        chunks: list[dict],
        video_chapters: list[dict] | None = None,
    ) -> ChunkingMetricResult:
        """
        chunks: list of TranscriptChunk dataclass converted to dict
                (or dicts from Pinecone metadata).
        video_chapters: list of chapter dicts from VideoMeta.
        """
        required = [
            "chunk_id", "video_id", "course_id", "chunk_index",
            "start_sec", "end_sec", "raw_text", "topic",
            "concept_tags", "chapter_title",
        ]
        critical = [
            "chunk_id", "video_id", "course_id", "raw_text",
        ]
        meta = _metadata_completeness(chunks, required, critical)

        token_counts = [_approx_tokens(c.get("raw_text", "")) for c in chunks]
        size_stats = _chunk_size_stats(token_counts, min_limit=60, max_limit=550)

        # Chapter alignment rate
        chapter_aligned = sum(
            1 for c in chunks
            if c.get("chapter_title") and c.get("chapter_title") != "Unknown"
        )
        n = len(chunks) or 1

        # Time continuity: no backwards timestamps
        sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
        time_gaps = []
        for i in range(1, len(sorted_chunks)):
            prev_end = sorted_chunks[i - 1].get("end_sec", 0)
            curr_start = sorted_chunks[i].get("start_sec", 0)
            time_gaps.append(curr_start - prev_end)

        overlap_rate = sum(1 for g in time_gaps if g < -1.0) / max(len(time_gaps), 1)

        extras = {
            "chapter_alignment_rate": round(chapter_aligned / n, 4),
            "avg_concept_tags_per_chunk": round(
                statistics.mean([len(c.get("concept_tags", [])) for c in chunks]) if chunks else 0, 2
            ),
            "temporal_overlap_rate": round(overlap_rate, 4),  # should be ~0
            "has_deep_link_rate": round(
                sum(1 for c in chunks if c.get("deep_link")) / n, 4
            ),
            "manual_transcript_rate": round(
                sum(1 for c in chunks if not c.get("is_generated_transcript")) / n, 4
            ),
        }

        return ChunkingMetricResult(
            pipeline="youtube",
            chunk_size_stats=size_stats,
            metadata_completeness=meta,
            extras=extras,
        )