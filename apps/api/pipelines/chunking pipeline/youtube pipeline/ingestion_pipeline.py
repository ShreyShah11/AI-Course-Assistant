"""
youtube_ingestion.py
====================
Ingestion pipeline for YouTube educational videos.

Flow:
    YouTube URL
    → Video metadata (pytubefix: title, author, chapters, keywords, duration …)
    → Transcript (youtube-transcript-api==1.2.4)
    → Chunking  (chapter-aware → semantic → timed fallback)
    → Deterministic metadata  (summary + topic + concept tags)
    → Gemini Embedding  (project-standard embedding model/dimension)
    → Pinecone  (index=<course_id>, namespace="youtube-chunks")

Install:
    pip install youtube-transcript-api==1.2.4 pytubefix google-genai \
                pinecone-client python-dotenv tqdm
"""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_REGION  = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

EMBEDDING_MODEL  = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM    = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))

PINECONE_NAMESPACE = os.getenv("YOUTUBE_CHUNKING_NAMESPACE", "youtube-chunks")
PINECONE_CLOUD     = os.getenv("PINECONE_CLOUD", "aws")

# Chunking constants
TARGET_CHUNK_WORDS  = 300   # ~2-3 minutes of speech @ avg lecture pace
MAX_CHUNK_WORDS     = 450   # hard cap before forced split
MIN_CHUNK_WORDS     = 80    # below this, merge with neighbour
OVERLAP_WORDS       = 40    # trailing-word overlap between chunks

# Rate limits
EMBED_SLEEP_SEC = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoMeta:
    """Metadata scraped from YouTube for a single video."""
    video_id:      str
    url:           str
    title:         str
    author:        str           # channel / professor name
    channel_url:   str
    duration_sec:  int
    publish_date:  str           # ISO date string or empty
    description:   str
    thumbnail_url: str
    yt_keywords:   list[str]     # tags the uploader set
    chapters:      list[dict]    # [{title, start_sec, duration_sec}]
    is_generated_transcript: bool
    transcript_language: str


@dataclass
class TranscriptChunk:
    """One semantic chunk ready for embedding and storage."""
    chunk_id:        str
    video_id:        str
    course_id:       str
    course_name:     str

    # Position
    chunk_index:     int          # 0-based across full video
    start_sec:       float
    end_sec:         float
    start_label:     str          # "MM:SS"
    end_label:       str

    # Content
    raw_text:        str          # verbatim transcript text for this chunk
    summary:         str          # 1-2 sentence LLM summary
    topic:           str          # short heading
    concept_tags:    list[str]    # domain concepts (e.g. "gradient descent")

    # Chapter context (if video has chapters)
    chapter_title:   str
    chapter_index:   int

    # Video-level metadata (denormalised for retrieval)
    video_title:     str
    author:          str
    channel_url:     str
    duration_sec:    int
    publish_date:    str
    thumbnail_url:   str
    yt_keywords:     list[str]
    is_generated_transcript: bool
    transcript_language: str

    # Course context
    instructor:      str
    semester:        str
    subject:         str
    tags:            list[str]
    extra:           dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Convert seconds → MM:SS (or HH:MM:SS for long videos)."""
    td = timedelta(seconds=int(seconds))
    total_sec = int(td.total_seconds())
    h, rem = divmod(total_sec, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _extract_video_id(url: str) -> str:
    """Parse video ID from any standard YouTube URL format."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    # Maybe raw ID was passed directly
    if re.match(r"^[A-Za-z0-9_-]{11}$", url.strip()):
        return url.strip()
    raise ValueError(f"Could not extract video ID from: {url!r}")


def _word_count(text: str) -> int:
    return len(text.split())


def _sanitize_index_name(course_id: str) -> str:
    name = re.sub(r"[^a-z0-9\-]", "-", course_id.lower())
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name[:45]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Fetch video metadata
# ─────────────────────────────────────────────────────────────────────────────

def fetch_video_metadata(video_id: str, url: str) -> VideoMeta:
    """
    Pull rich metadata from YouTube using pytubefix.
    Falls back gracefully for any unavailable field.
    """
    try:
        from pytubefix import YouTube
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

        # Chapters (gold for educational videos — lectures often have them)
        chapters_raw = []
        try:
            for ch in (yt.chapters or []):
                chapters_raw.append({
                    "title":       ch.title,
                    "start_sec":   ch.start_seconds,
                    "duration_sec": ch.duration,
                    "end_sec":     ch.start_seconds + ch.duration,
                })
        except Exception:
            pass

        publish_date = ""
        try:
            if yt.publish_date:
                publish_date = yt.publish_date.isoformat()
        except Exception:
            pass

        return VideoMeta(
            video_id      = video_id,
            url           = url,
            title         = yt.title or "",
            author        = yt.author or "",
            channel_url   = yt.channel_url or "",
            duration_sec  = yt.length or 0,
            publish_date  = publish_date,
            description   = (yt.description or "")[:2000],  # cap for storage
            thumbnail_url = yt.thumbnail_url or "",
            yt_keywords   = list(yt.keywords or []),
            chapters      = chapters_raw,
            is_generated_transcript=False,  # filled later
            transcript_language="",         # filled later
        )

    except Exception as exc:
        print(f"  [WARN] pytubefix metadata fetch failed: {exc}. Using minimal meta.")
        return VideoMeta(
            video_id=video_id, url=url, title="", author="", channel_url="",
            duration_sec=0, publish_date="", description="", thumbnail_url="",
            yt_keywords=[], chapters=[], is_generated_transcript=False,
            transcript_language="",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Fetch transcript
# ─────────────────────────────────────────────────────────────────────────────

def fetch_transcript(
    video_id: str,
    languages: list[str] = ("en",),
) -> tuple[list[dict], bool, str]:
    """
    Fetch transcript using youtube-transcript-api==1.2.4.

    Returns (snippets, is_generated, language_code)
    where snippets = [{"text": str, "start": float, "duration": float}, …]

    Strategy:
    1. Try manual captions in preferred languages first (more accurate).
    2. Fall back to auto-generated captions.
    3. If nothing found, raise.
    """
    api = YouTubeTranscriptApi()

    # Try manual captions first
    try:
        transcript_list = api.list(video_id)

        # Prefer manually created in given language order
        for lang in languages:
            try:
                t = transcript_list._manually_created_transcripts.get(lang)
                if t:
                    fetched = t.fetch()
                    return fetched.to_raw_data(), False, fetched.language_code
            except Exception:
                pass

        # Fallback: auto-generated
        for lang in languages:
            try:
                t = transcript_list._generated_transcripts.get(lang)
                if t:
                    fetched = t.fetch()
                    return fetched.to_raw_data(), True, fetched.language_code
            except Exception:
                pass

    except (TranscriptsDisabled, VideoUnavailable):
        raise

    # Final fallback: fetch() with language preference (raises if nothing found)
    fetched = api.fetch(video_id, languages=list(languages))
    return fetched.to_raw_data(), fetched.is_generated, fetched.language_code


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Chunking  (chapter-aware → semantic → timed fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _assign_chapter(start_sec: float, chapters: list[dict]) -> tuple[str, int]:
    """Return (chapter_title, chapter_index) for a snippet at start_sec."""
    if not chapters:
        return "Unknown", -1
    assigned_title, assigned_idx = chapters[0]["title"], 0
    for i, ch in enumerate(chapters):
        if start_sec >= ch["start_sec"]:
            assigned_title, assigned_idx = ch["title"], i
        else:
            break
    return assigned_title, assigned_idx


def _build_raw_chunks_chapter_aware(
    snippets: list[dict],
    chapters: list[dict],
) -> list[dict]:
    """
    PRIMARY strategy: if video has chapters, respect chapter boundaries.
    Within each chapter, accumulate snippets up to TARGET_CHUNK_WORDS,
    then split. This keeps related content together perfectly.
    """
    # Group snippets by chapter
    chapter_buckets: dict[int, list[dict]] = {}
    for snip in snippets:
        _, ch_idx = _assign_chapter(snip["start"], chapters)
        chapter_buckets.setdefault(ch_idx, []).append(snip)

    raw_chunks = []
    for ch_idx in sorted(chapter_buckets.keys()):
        ch_snippets = chapter_buckets[ch_idx]
        ch_title    = chapters[ch_idx]["title"] if ch_idx < len(chapters) else "Unknown"

        # Split chapter into word-count chunks
        current_words = []
        current_snips = []
        for snip in ch_snippets:
            words = snip["text"].split()
            current_words.extend(words)
            current_snips.append(snip)

            if len(current_words) >= TARGET_CHUNK_WORDS:
                raw_chunks.append({
                    "chapter_title": ch_title,
                    "chapter_index": ch_idx,
                    "snippets":      current_snips,
                    "text":          " ".join(current_words),
                })
                # Carry overlap forward
                overlap_text  = current_words[-OVERLAP_WORDS:]
                current_words = list(overlap_text)
                current_snips = current_snips[-3:]  # keep last few snippets

        # Flush remainder
        if current_words and _word_count(" ".join(current_words)) >= MIN_CHUNK_WORDS:
            raw_chunks.append({
                "chapter_title": ch_title,
                "chapter_index": ch_idx,
                "snippets":      current_snips,
                "text":          " ".join(current_words),
            })
        elif current_words and raw_chunks:
            # Too short — merge into previous chunk
            raw_chunks[-1]["text"] += " " + " ".join(current_words)
            raw_chunks[-1]["snippets"].extend(current_snips)

    return raw_chunks


def _build_raw_chunks_timed(snippets: list[dict]) -> list[dict]:
    """
    FALLBACK strategy: no chapters available.
    Slide over snippets by word count with overlap.
    Looks for sentence-ending boundaries near the target to make cleaner cuts.
    """
    raw_chunks = []
    current_words:  list[str] = []
    current_snips:  list[dict] = []

    for snip in snippets:
        words = snip["text"].split()
        current_words.extend(words)
        current_snips.append(snip)

        if len(current_words) >= TARGET_CHUNK_WORDS:
            # Try to cut at a sentence boundary near the target
            cut_text = " ".join(current_words)
            # Find last sentence-ending punctuation near target
            approx_char = int(len(cut_text) * (TARGET_CHUNK_WORDS / len(current_words)))
            search_window = cut_text[max(0, approx_char - 80): approx_char + 80]
            m = list(re.finditer(r'[.!?]\s', search_window))
            if m:
                boundary_offset = max(0, approx_char - 80) + m[-1].end()
                final_text = cut_text[:boundary_offset].strip()
            else:
                final_text = cut_text

            raw_chunks.append({
                "chapter_title": "Unknown",
                "chapter_index": -1,
                "snippets":      current_snips,
                "text":          final_text,
            })
            # Overlap carry-forward
            current_words = current_words[-OVERLAP_WORDS:]
            current_snips = current_snips[-3:]

    # Flush
    if current_words:
        leftover_text = " ".join(current_words)
        if _word_count(leftover_text) >= MIN_CHUNK_WORDS:
            raw_chunks.append({
                "chapter_title": "Unknown",
                "chapter_index": -1,
                "snippets":      current_snips,
                "text":          leftover_text,
            })
        elif raw_chunks:
            raw_chunks[-1]["text"] += " " + leftover_text

    return raw_chunks


def chunk_transcript(
    snippets: list[dict],
    chapters: list[dict],
) -> list[dict]:
    """
    Master chunking dispatcher.
    Returns list of raw chunk dicts with text, timing, chapter info.
    """
    if chapters:
        print(f"      → Chapter-aware chunking ({len(chapters)} chapters detected)")
        raw = _build_raw_chunks_chapter_aware(snippets, chapters)
    else:
        print("      → Timed fallback chunking (no chapters)")
        raw = _build_raw_chunks_timed(snippets)

    # Attach timing from first/last snippet in each chunk
    for rc in raw:
        snips = rc["snippets"]
        if snips:
            rc["start_sec"] = snips[0]["start"]
            last = snips[-1]
            rc["end_sec"] = last["start"] + last.get("duration", 0)
        else:
            rc["start_sec"] = 0.0
            rc["end_sec"]   = 0.0

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Deterministic metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "between", "could",
    "during", "example", "following", "from", "have", "into", "like", "more",
    "most", "only", "other", "should", "that", "their", "there", "these",
    "they", "this", "through", "what", "when", "where", "which", "while",
    "with", "would",
}


def _keyword_tags(text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z_-]{3,}\b", text.lower())
    freq: dict[str, int] = {}
    for token in tokens:
        if token not in _STOPWORDS:
            freq[token] = freq.get(token, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:limit]


def _deterministic_chunk_metadata(
    text: str,
    video_title: str,
    chapter_title: str,
) -> dict:
    tags = _keyword_tags(text)
    topic = chapter_title if chapter_title and chapter_title != "Unknown" else video_title
    topic = (topic or "YouTube Lecture").strip()[:80]
    words = text.split()
    summary = " ".join(words[:70]).strip()
    if len(words) > 70:
        summary += "..."
    return {
        "topic": topic,
        "summary": summary,
        "concept_tags": tags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Embed
# ─────────────────────────────────────────────────────────────────────────────

def _embed_text(text: str, retries: int = 3) -> list[float]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    for attempt in range(1, retries + 1):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[text if text.strip() else " "],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return result.embeddings[0].values
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Embedding failed: {exc}") from exc
            time.sleep(2 ** attempt)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Pinecone
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_index(pc: Pinecone, index_name: str):
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"  [Pinecone] Creating index '{index_name}' …")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        for _ in range(20):
            if pc.describe_index(index_name).status.get("ready", False):
                break
            time.sleep(3)
    return pc.Index(index_name)


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def ingest_youtube_video(
    url: str,
    course_id: str,
    course_name: str,
    instructor: str = "",
    semester: str = "",
    subject: str = "",
    tags: list[str] | None = None,
    languages: list[str] | None = None,
    batch_size: int = 50,
    index_name: str | None = None,
    namespace: str | None = None,
) -> list[TranscriptChunk]:
    """
    Full ingestion pipeline for a YouTube educational video.

    Parameters
    ----------
    url          : any valid YouTube URL or 11-char video ID
    course_id    : used as Pinecone index name
    course_name  : stored in chunk metadata
    instructor   : optional professor/instructor name
    semester     : optional e.g. "Fall 2024"
    subject      : optional subject area
    tags         : optional custom filter tags
    languages    : transcript language preference order (default ["en"])
    batch_size   : Pinecone upsert batch size
    """
    tags = [tag for tag in (tags or []) if tag and tag != "string"]
    languages = [lang for lang in (languages or ["en"]) if lang and lang != "string"] or ["en"]

    print(f"\n{'='*65}")
    print(f"  YouTube Lecture Ingestion Pipeline")
    print(f"  URL    : {url}")
    print(f"  Course : {course_id} — {course_name}")
    print(f"{'='*65}\n")

    # ── Init ────────────────────────────────────────────────────────────────
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")

    pc         = Pinecone(api_key=PINECONE_API_KEY)
    index_name = index_name or _sanitize_index_name(course_id)
    namespace = namespace or PINECONE_NAMESPACE
    index = _get_or_create_index(pc, index_name)

    # ── 1. Video ID ──────────────────────────────────────────────────────────
    video_id = _extract_video_id(url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"[1/6] Video ID: {video_id}")

    # ── 2. Metadata ──────────────────────────────────────────────────────────
    print(f"[2/6] Fetching video metadata …")
    meta = fetch_video_metadata(video_id, canonical_url)
    duration_str = _fmt_time(meta.duration_sec)
    print(f"      Title    : {meta.title or '(unknown)'}")
    print(f"      Author   : {meta.author or '(unknown)'}")
    print(f"      Duration : {duration_str}")
    print(f"      Chapters : {len(meta.chapters)}")

    # ── 3. Transcript ────────────────────────────────────────────────────────
    print(f"[3/6] Fetching transcript (languages={languages}) …")
    try:
        snippets, is_generated, lang_code = fetch_transcript(video_id, languages)
    except TranscriptsDisabled:
        print("  [ERROR] Transcripts are disabled for this video.")
        return []
    except NoTranscriptFound:
        print(f"  [ERROR] No transcript found in languages {languages}.")
        return []

    meta.is_generated_transcript = is_generated
    meta.transcript_language     = lang_code
    total_words = sum(_word_count(s["text"]) for s in snippets)
    print(f"      → {len(snippets)} snippets  |  ~{total_words} words  "
          f"|  {'auto-generated' if is_generated else 'manual'} ({lang_code})")

    # ── 4. Chunking ──────────────────────────────────────────────────────────
    print(f"[4/6] Chunking transcript …")
    raw_chunks = chunk_transcript(snippets, meta.chapters)
    print(f"      → {len(raw_chunks)} chunk(s) produced.\n")

    if not raw_chunks:
        print("[WARN] No chunks produced. Aborting.")
        return []

    # ── 5. Metadata + embedding ─────────────────────────────────────────────
    print(f"[5/6] Preparing metadata & embedding {len(raw_chunks)} chunk(s) …")
    chunks: list[TranscriptChunk] = []
    vectors: list[dict] = []

    for idx, rc in enumerate(tqdm(raw_chunks, desc="Enrich+Embed")):
        start_label = _fmt_time(rc["start_sec"])
        end_label   = _fmt_time(rc["end_sec"])

        enriched = _deterministic_chunk_metadata(
            text          = rc["text"],
            video_title   = meta.title,
            chapter_title = rc["chapter_title"],
        )

        # Build chunk object
        chunk = TranscriptChunk(
            chunk_id     = str(uuid.uuid4()),
            video_id     = video_id,
            course_id    = course_id,
            course_name  = course_name,

            chunk_index  = idx,
            start_sec    = rc["start_sec"],
            end_sec      = rc["end_sec"],
            start_label  = start_label,
            end_label    = end_label,

            raw_text     = rc["text"],
            summary      = enriched["summary"],
            topic        = enriched["topic"],
            concept_tags = enriched["concept_tags"],

            chapter_title = rc["chapter_title"],
            chapter_index = rc["chapter_index"],

            video_title  = meta.title,
            author       = meta.author,
            channel_url  = meta.channel_url,
            duration_sec = meta.duration_sec,
            publish_date = meta.publish_date,
            thumbnail_url= meta.thumbnail_url,
            yt_keywords  = meta.yt_keywords,
            is_generated_transcript = meta.is_generated_transcript,
            transcript_language     = meta.transcript_language,

            instructor   = instructor,
            semester     = semester,
            subject      = subject,
            tags         = tags,
        )
        chunks.append(chunk)

        # Deep embed: topic + summary + raw_text  (follows handwritten pipeline convention)
        embed_input = f"{chunk.topic}\n{chunk.summary}\n{chunk.raw_text}"
        embedding   = _embed_text(embed_input)
        time.sleep(EMBED_SLEEP_SEC)

        # Build Pinecone vector
        # Note: deep-link URL lets retrieval surface exact timestamp
        deep_link = f"https://www.youtube.com/watch?v={video_id}&t={int(rc['start_sec'])}s"

        metadata = {
            # Identity
            "chunk_id":      chunk.chunk_id,
            "video_id":      chunk.video_id,
            "course_id":     chunk.course_id,
            "course_name":   chunk.course_name,

            # Position
            "chunk_index":   chunk.chunk_index,
            "start_sec":     chunk.start_sec,
            "end_sec":       chunk.end_sec,
            "start_label":   chunk.start_label,
            "end_label":     chunk.end_label,
            "deep_link":     deep_link,

            # Content
            "raw_text":      chunk.raw_text,
            "summary":       chunk.summary,
            "topic":         chunk.topic,
            "concept_tags":  chunk.concept_tags,

            # Chapter
            "chapter_title": chunk.chapter_title,
            "chapter_index": chunk.chapter_index,

            # Video-level
            "video_title":   chunk.video_title,
            "author":        chunk.author,
            "channel_url":   chunk.channel_url,
            "duration_sec":  chunk.duration_sec,
            "publish_date":  chunk.publish_date,
            "thumbnail_url": chunk.thumbnail_url,
            "yt_keywords":   chunk.yt_keywords,
            "is_generated_transcript": chunk.is_generated_transcript,
            "transcript_language":     chunk.transcript_language,

            # Course context
            "instructor":    chunk.instructor,
            "semester":      chunk.semester,
            "subject":       chunk.subject,
            "tags":          chunk.tags,
        }

        vectors.append({"id": chunk.chunk_id, "values": embedding, "metadata": metadata})

    # ── 6. Upsert ────────────────────────────────────────────────────────────
    print(f"\n[6/6] Upserting {len(vectors)} vector(s) to Pinecone …")
    print(f"      Index     : {index_name}")
    print(f"      Namespace : {namespace}")

    for i in tqdm(range(0, len(vectors), batch_size), desc="Upsert batches"):
        index.upsert(vectors=vectors[i: i + batch_size], namespace=namespace)

    print(f"\n✅ Ingestion complete!")
    print(f"   Video    : {meta.title}")
    print(f"   Chunks   : {len(chunks)}")
    print(f"   Index    : {index_name}  |  namespace: {namespace}\n")

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a YouTube educational video into Pinecone."
    )
    parser.add_argument("url",           help="YouTube URL or video ID")
    parser.add_argument("--course-id",   required=True)
    parser.add_argument("--course-name", required=True)
    parser.add_argument("--instructor",  default="")
    parser.add_argument("--semester",    default="")
    parser.add_argument("--subject",     default="")
    parser.add_argument("--tags",        nargs="*", default=[])
    parser.add_argument("--languages",   nargs="*", default=["en"],
                        help="Transcript language preference order")
    args = parser.parse_args()

    chunks = ingest_youtube_video(
        url         = args.url,
        course_id   = args.course_id,
        course_name = args.course_name,
        instructor  = args.instructor,
        semester    = args.semester,
        subject     = args.subject,
        tags        = args.tags,
        languages   = args.languages,
    )

    if not chunks:
        return

    # Summary table
    print(f"\n{'─'*80}")
    print(f"{'IDX':>4}  {'TIME':>10}  {'CHAPTER':<25}  {'TOPIC':<28}  WORDS")
    print(f"{'─'*80}")
    for c in chunks:
        wc = _word_count(c.raw_text)
        print(f"{c.chunk_index:>4}  {c.start_label:>5}→{c.end_label:<5}  "
              f"{c.chapter_title:<25.25}  {c.topic:<28.28}  {wc}")
    print(f"{'─'*80}\n")


if __name__ == "__main__":
    main()
