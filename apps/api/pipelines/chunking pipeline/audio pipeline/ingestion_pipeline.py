"""
Course Assistant Chunking Pipeline
====================================
Designed for: University AI course assistant
  - Input  : long professor monologues (45–90 min lectures)
  - Queries: concept / fact questions with no lecture reference
             e.g. "explain dynamic programming", "what is osmosis"
  - Index  : one vector DB per course, all lectures stored together

Three complementary strategies
--------------------------------
1. sliding_window   – overlapping fixed-size windows (retrieval backbone)
2. concept_block    – variable-size blocks around a single concept/topic
                      (best match for "explain X" queries)
3. lecture_summary  – one macro-chunk per lecture (coarse retrieval layer)

Each chunk carries full lecture + course provenance so you can pre-filter
by course_id / lecture_id before the ANN search.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

from generate_transcripts import TranscriptMetadata, TranscriptSegment


# ──────────────────────────────────────────────────────────────────────────────
# Lecture identity  (caller must supply this)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LectureIdentity:
    """
    Attach this to every transcription before chunking.
    Stored verbatim in chunk metadata — use for pre-filtering in your vector DB.

    Example
    -------
    identity = LectureIdentity(
        course_id   = "CS301",
        course_name = "Algorithms & Data Structures",
        lecture_id  = "CS301_L07",
        lecture_number = 7,
        week_number    = 4,
        lecture_title  = "Dynamic Programming – Part 1",
    )
    """
    course_id: str
    course_name: str
    lecture_id: str                     # must be unique within the course index
    lecture_number: int
    week_number: int
    lecture_title: str = ""
    professor: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Chunk schema
# ──────────────────────────────────────────────────────────────────────────────

class ChunkStrategy(str, Enum):
    SLIDING_WINDOW  = "sliding_window"
    CONCEPT_BLOCK   = "concept_block"
    LECTURE_SUMMARY = "lecture_summary"


@dataclass
class CourseChunk:
    # ── identity ──────────────────────────────────────────────────────────────
    chunk_id: str                       # deterministic 24-char SHA-256

    # ── course / lecture provenance (pre-filter keys) ─────────────────────────
    course_id: str
    course_name: str
    lecture_id: str
    lecture_number: int
    week_number: int
    lecture_title: str
    professor: str

    # ── content ───────────────────────────────────────────────────────────────
    strategy: str                       # ChunkStrategy value
    text: str                           # ← embed this
    word_count: int

    # ── temporal ──────────────────────────────────────────────────────────────
    start_seconds: float
    end_seconds: float
    duration_seconds: float

    # ── segment membership ────────────────────────────────────────────────────
    segment_indices: list[int]
    segment_count: int

    # ── concept / topic signals ───────────────────────────────────────────────
    concepts: list[str]                 # active topics in this chunk
    keywords: list[str]                 # searchable keywords from TranscriptMetadata
    is_concept_boundary: bool           # True if this chunk starts a new concept
    concept_label: str                  # short label for the main concept (if detected)

    # ── quality signals ───────────────────────────────────────────────────────
    avg_confidence: float
    min_confidence: float
    has_unclear_segments: bool

    # ── window bookkeeping ────────────────────────────────────────────────────
    window_index: int = 0
    overlap_with_prev: int = 0          # segments shared with previous chunk
    overlap_with_next: int = 0          # backfilled after all chunks are created

    # ── source provenance ─────────────────────────────────────────────────────
    source_file: str = ""
    file_name: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_id(lecture_id: str, strategy: str, window_index: int) -> str:
    key = f"{lecture_id}|{strategy}|{window_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _confidences(segments: list[TranscriptSegment]) -> tuple[float, float]:
    vals = [s.confidence for s in segments]
    if not vals:
        return 0.0, 0.0
    return round(sum(vals) / len(vals), 4), round(min(vals), 4)


def _active_concepts(
    segments: list[TranscriptSegment],
    meta: TranscriptMetadata,
) -> list[str]:
    """
    Topics from TranscriptMetadata whose keywords appear in the chunk text.
    Falls back to all topics so the field is never empty.
    """
    chunk_lower = " ".join(s.text for s in segments).lower()
    matched = [t for t in meta.topics if t.lower() in chunk_lower]
    return matched if matched else list(meta.topics)


def _concept_label(concepts: list[str], segments: list[TranscriptSegment]) -> str:
    """Best single-label for this chunk (most-frequent topic keyword in text)."""
    if not concepts:
        return ""
    text_lower = " ".join(s.text for s in segments).lower()
    return max(concepts, key=lambda t: text_lower.count(t.lower()))


def _backfill_overlap_next(chunks: list[CourseChunk]) -> None:
    for i, chunk in enumerate(chunks[:-1]):
        chunk.overlap_with_next = chunks[i + 1].overlap_with_prev


def _build_chunk(
    meta: TranscriptMetadata,
    identity: LectureIdentity,
    segments: list[TranscriptSegment],
    strategy: ChunkStrategy,
    window_index: int,
    overlap_prev: int = 0,
    is_concept_boundary: bool = False,
) -> CourseChunk:
    text = " ".join(s.text.strip() for s in segments)
    avg_conf, min_conf = _confidences(segments)
    concepts = _active_concepts(segments, meta)
    start = segments[0].start_seconds
    end = segments[-1].end_seconds

    return CourseChunk(
        chunk_id=_chunk_id(identity.lecture_id, strategy.value, window_index),
        course_id=identity.course_id,
        course_name=identity.course_name,
        lecture_id=identity.lecture_id,
        lecture_number=identity.lecture_number,
        week_number=identity.week_number,
        lecture_title=identity.lecture_title,
        professor=identity.professor,
        strategy=strategy.value,
        text=text,
        word_count=_word_count(text),
        start_seconds=start,
        end_seconds=end,
        duration_seconds=max(end - start, 0.0),
        segment_indices=[s.segment_index for s in segments],
        segment_count=len(segments),
        concepts=concepts,
        keywords=meta.searchable_keywords,
        is_concept_boundary=is_concept_boundary,
        concept_label=_concept_label(concepts, segments),
        avg_confidence=avg_conf,
        min_confidence=min_conf,
        has_unclear_segments=any(s.is_unclear for s in segments),
        window_index=window_index,
        overlap_with_prev=overlap_prev,
        source_file=meta.source_file,
        file_name=meta.file_name,
        model=meta.model,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Sliding Window
# ──────────────────────────────────────────────────────────────────────────────

def chunk_sliding_window(
    meta: TranscriptMetadata,
    identity: LectureIdentity,
    window_size: int = 6,
    overlap: int = 2,
) -> list[CourseChunk]:
    """
    Overlapping fixed-size windows over segments.

    This is the retrieval BACKBONE — it guarantees no content is unreachable.
    Overlap ensures a concept that spans a window boundary still appears in
    at least one complete chunk.

    Tuning for 45–90 min lectures
    ─────────────────────────────
    Each TranscriptSegment ≈ 10–20 s  →  6 segments ≈ 60–120 s per chunk.
    That's one coherent explanation block — not too wide (noisy), not too
    narrow (loses context).

    window_size=6, overlap=2  →  step=4  (default, recommended)
    window_size=8, overlap=3  →  for very dense lectures with long explanations
    window_size=4, overlap=1  →  for faster-paced lectures with short topics
    """
    segs = meta.segments
    if not segs:
        return []

    step = max(1, window_size - overlap)
    chunks: list[CourseChunk] = []
    wi = 0

    for start_idx in range(0, len(segs), step):
        window = segs[start_idx: start_idx + window_size]
        if not window:
            break
        overlap_prev = overlap if start_idx > 0 else 0
        chunks.append(_build_chunk(
            meta, identity, window,
            ChunkStrategy.SLIDING_WINDOW, wi, overlap_prev,
        ))
        wi += 1
        if start_idx + window_size >= len(segs):
            break

    _backfill_overlap_next(chunks)
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Concept Block
# ──────────────────────────────────────────────────────────────────────────────

def _detect_concept_boundaries(
    segs: list[TranscriptSegment],
    meta: TranscriptMetadata,
    min_block_segments: int,
    max_block_segments: int,
) -> list[int]:
    """
    Returns indices in `segs` where a new concept starts.

    Detection logic (no LLM call, runs on TranscriptMetadata already produced):
    A boundary fires when the dominant topic keyword in the rolling window
    changes AND we've accumulated at least min_block_segments since the
    last boundary.  Hard cap at max_block_segments prevents runaway blocks.
    """
    if not meta.topics or len(segs) < 2:
        # No topic info — fall back to even splits
        return list(range(0, len(segs), min_block_segments))

    boundaries = [0]
    last_boundary = 0

    def dominant_topic(window: list[TranscriptSegment]) -> str:
        text = " ".join(s.text for s in window).lower()
        scored = {t: text.count(t.lower()) for t in meta.topics}
        best = max(scored, key=scored.get)
        return best if scored[best] > 0 else "__none__"

    window_size = max(3, min_block_segments)
    current_topic = dominant_topic(segs[:window_size])

    for i in range(1, len(segs)):
        segments_since = i - last_boundary
        # Hard cap — force a boundary
        if segments_since >= max_block_segments:
            boundaries.append(i)
            last_boundary = i
            current_topic = dominant_topic(
                segs[i: i + window_size]
            )
            continue

        if segments_since < min_block_segments:
            continue

        # Soft boundary — topic shift detected
        window = segs[i: i + window_size]
        new_topic = dominant_topic(window)
        if new_topic != current_topic and new_topic != "__none__":
            boundaries.append(i)
            last_boundary = i
            current_topic = new_topic

    return boundaries


def chunk_concept_blocks(
    meta: TranscriptMetadata,
    identity: LectureIdentity,
    min_block_segments: int = 4,
    max_block_segments: int = 12,
) -> list[CourseChunk]:
    """
    Variable-length blocks, each covering one concept or topic.

    This is the PRIMARY strategy for "explain X" / "what is Y" queries.
    The block boundaries are detected from the topics list produced by
    Gemini during transcription — zero extra LLM cost.

    Why this beats fixed windows for concept queries
    ────────────────────────────────────────────────
    A professor explains 'recursion' for ~8 segments then shifts to
    'memoization' for ~5 segments. A fixed window of 6 would split the
    recursion explanation in half. A concept block keeps it whole.

    Tuning
    ──────
    min_block_segments=4   →  blocks ≥ ~40–80 s  (avoids tiny transitional chunks)
    max_block_segments=12  →  blocks ≤ ~2–4 min  (caps runaway monologues)
    """
    segs = meta.segments
    if not segs:
        return []

    boundaries = _detect_concept_boundaries(
        segs, meta, min_block_segments, max_block_segments
    )
    # Add sentinel
    boundaries.append(len(segs))

    chunks: list[CourseChunk] = []
    for wi, (start_idx, end_idx) in enumerate(zip(boundaries, boundaries[1:])):
        block = segs[start_idx:end_idx]
        if not block:
            continue
        # Mark first chunk and any chunk starting after a topic shift as a boundary
        is_boundary = wi == 0 or True  # all concept blocks are boundaries by definition
        chunks.append(_build_chunk(
            meta, identity, block,
            ChunkStrategy.CONCEPT_BLOCK, wi,
            overlap_prev=0,
            is_concept_boundary=True,
        ))

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 3 — Lecture Summary
# ──────────────────────────────────────────────────────────────────────────────

def chunk_lecture_summary(
    meta: TranscriptMetadata,
    identity: LectureIdentity,
) -> list[CourseChunk]:
    """
    One macro-chunk per lecture using the summary + topics from TranscriptMetadata.

    Role in retrieval
    ─────────────────
    Acts as a COARSE retrieval layer.  When a student's question is broad
    ("what did we cover in algorithms this week?") the summary chunk scores
    highly.  For narrow questions, it scores low and the sliding/concept chunks
    win — that's correct behaviour.

    In a two-stage retrieval setup:
      1. Fetch top-K from all strategies (summary + sliding + concept)
      2. Rerank — summary chunks provide lecture-level context to the LLM
         while concept/sliding chunks provide the actual explanation text.
    """
    if not meta.segments:
        return []

    # Build a rich summary text: summary prose + bulleted topics + keywords
    topic_list = "\n".join(f"- {t}" for t in meta.topics) if meta.topics else ""
    keyword_str = ", ".join(meta.searchable_keywords) if meta.searchable_keywords else ""

    text_parts = [f"Lecture summary: {meta.summary}"]
    if topic_list:
        text_parts.append(f"Topics covered:\n{topic_list}")
    if keyword_str:
        text_parts.append(f"Key terms: {keyword_str}")
    text = "\n\n".join(text_parts)

    avg_conf, min_conf = _confidences(meta.segments)

    chunk = CourseChunk(
        chunk_id=_chunk_id(identity.lecture_id, ChunkStrategy.LECTURE_SUMMARY.value, 0),
        course_id=identity.course_id,
        course_name=identity.course_name,
        lecture_id=identity.lecture_id,
        lecture_number=identity.lecture_number,
        week_number=identity.week_number,
        lecture_title=identity.lecture_title,
        professor=identity.professor,
        strategy=ChunkStrategy.LECTURE_SUMMARY.value,
        text=text,
        word_count=_word_count(text),
        start_seconds=meta.segments[0].start_seconds,
        end_seconds=meta.segments[-1].end_seconds,
        duration_seconds=meta.segments[-1].end_seconds - meta.segments[0].start_seconds,
        segment_indices=list(range(len(meta.segments))),
        segment_count=len(meta.segments),
        concepts=list(meta.topics),
        keywords=meta.searchable_keywords,
        is_concept_boundary=True,
        concept_label=meta.topics[0] if meta.topics else "",
        avg_confidence=avg_conf,
        min_confidence=min_conf,
        has_unclear_segments=any(s.is_unclear for s in meta.segments),
        window_index=0,
        source_file=meta.source_file,
        file_name=meta.file_name,
        model=meta.model,
    )
    return [chunk]


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline config + entry point
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CourseChunkingConfig:
    """All tuning knobs in one place."""
    # sliding window
    sliding_window_size: int = 6        # segments per chunk  (~60–120 s)
    sliding_overlap: int = 2            # shared segments between adjacent chunks

    # concept blocks
    concept_min_segments: int = 4       # minimum segments per concept block
    concept_max_segments: int = 12      # maximum segments per concept block


def run_pipeline(
    meta: TranscriptMetadata,
    identity: LectureIdentity,
    config: Optional[CourseChunkingConfig] = None,
) -> dict[str, list[CourseChunk]]:
    """
    Run all three strategies and return a dict keyed by strategy name.

    Returns
    -------
    {
        "sliding_window":  [...],
        "concept_block":   [...],
        "lecture_summary": [...],
    }
    """
    if config is None:
        config = CourseChunkingConfig()

    return {
        ChunkStrategy.SLIDING_WINDOW.value: chunk_sliding_window(
            meta, identity,
            window_size=config.sliding_window_size,
            overlap=config.sliding_overlap,
        ),
        ChunkStrategy.CONCEPT_BLOCK.value: chunk_concept_blocks(
            meta, identity,
            min_block_segments=config.concept_min_segments,
            max_block_segments=config.concept_max_segments,
        ),
        ChunkStrategy.LECTURE_SUMMARY.value: chunk_lecture_summary(
            meta, identity,
        ),
    }


def flatten_chunks(chunked: dict[str, list[CourseChunk]]) -> list[CourseChunk]:
    """All chunks from all strategies in one flat list."""
    return [c for chunks in chunked.values() for c in chunks]


# ──────────────────────────────────────────────────────────────────────────────
# Vector-store adapter
# ──────────────────────────────────────────────────────────────────────────────

def to_store_records(chunks: list[CourseChunk]) -> list[dict]:
    """
    Convert to the standard upsert format for Pinecone / Qdrant / pgvector /
    Weaviate / Chroma etc.

    Schema per record
    -----------------
    {
        "id":       chunk_id,
        "text":     chunk.text,          ← the field you embed
        "metadata": { ...everything else }
    }

    Pre-filter keys you can use in your vector DB query
    ---------------------------------------------------
    metadata.course_id       == "CS301"
    metadata.lecture_id      == "CS301_L07"
    metadata.week_number     <= 4
    metadata.strategy        == "concept_block"
    metadata.min_confidence  >= 0.75
    """
    records = []
    for c in chunks:
        d = c.to_dict()
        text = d.pop("text")
        chunk_id = d.pop("chunk_id")
        records.append({"id": chunk_id, "text": text, "metadata": d})
    return records
