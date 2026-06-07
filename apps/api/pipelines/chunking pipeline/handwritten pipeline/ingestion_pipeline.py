"""
handwritten_pipeline.py
========================
Self-contained handwritten notes ingestion pipeline.

Stages
------
1. Ingest    — PDF → rasterised PIL images, or load image directly
2. OCR       — Gemini 2.5 Flash per page; escalate to Pro if confidence low
3. Chunk     — semantic split on headings / paragraphs, merge tiny, split big
4. Embed     — Gemini embedding model, batched (RETRIEVAL_DOCUMENT task type)
5. Upsert    — Pinecone (auto-creates index); namespace = "handwritten-notes"

Quick start
-----------
    pip install google-genai pinecone Pillow pdf2image python-dotenv
    # macOS: brew install poppler  |  Ubuntu: sudo apt-get install poppler-utils

Set env vars (or create a .env file):
    GEMINI_API_KEY=...
    PINECONE_API_KEY=...
    PINECONE_INDEX_NAME=handwritten-notes   # optional, this is the default
    CONFIDENCE_THRESHOLD=0.75              # optional, this is the default

CLI usage
---------
    python handwritten_pipeline.py notes.pdf --course-id CS101 --course-name "Intro to CS"

    python handwritten_pipeline.py lecture3.pdf \\
        --course-id CS101 --course-name "Intro to CS" \\
        --subject "Computer Science" --instructor "Dr. Smith" \\
        --semester "Fall 2024" --university "MIT" \\
        --tags algorithms sorting big-o

    python handwritten_pipeline.py query "What is Big O?" --course-id CS101 --top-k 5

    python handwritten_pipeline.py stats

Python API
----------
    from handwritten_pipeline import run_pipeline, embed_query, query_notes

    result = run_pipeline(
        filepath="notes.pdf",
        course_id="CS101",
        course_name="Intro to Computer Science",
        subject="CS", instructor="Dr. Smith",
        semester="Fall 2024", university="MIT",
        tags=["algorithms"],
    )
    print(result.summary())

    results = query_notes(embed_query("explain recursion"), course_id="CS101", top_k=5)
    for r in results:
        print(r["score"], r["metadata"]["topic"])
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # API keys
    gemini_api_key: str   = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))

    # Pinecone
    pinecone_index_name: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "rag-index")
    )
    pinecone_namespace: str = field(
        default_factory=lambda: os.getenv("HANDWRITTEN_CHUNKING_NAMESPACE", "handwritten-chunks")
    )
    pinecone_dimension: int = field(
        default_factory=lambda: int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    )
    pinecone_cloud: str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(
        default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    )
    pinecone_metric: str     = "cosine"

    # Gemini models
    flash_model: str = field(
        default_factory=lambda: os.getenv("HANDWRITTEN_FLASH_MODEL", "gemini-2.5-flash")
    )
    pro_model: str = field(
        default_factory=lambda: os.getenv("HANDWRITTEN_PRO_MODEL", "gemini-2.5-pro")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    )

    # OCR escalation
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    )

    # Ingestor
    pdf_dpi: int       = 150
    max_image_px: int  = 2048   # long-edge cap before sending to Gemini

    # Chunker
    min_chunk_chars: int = 80
    max_chunk_chars: int = 2500

    # Embedder
    embed_batch_size: int = 100

    def validate(self) -> None:
        missing = [k for k, v in [
            ("GEMINI_API_KEY",  self.gemini_api_key),
            ("PINECONE_API_KEY", self.pinecone_api_key),
        ] if not v]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Set them in your shell or in a .env file."
            )


cfg = Config()
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "ocr.md"


class HandwrittenOCRResponse(BaseModel):
    transcript: str = ""
    structured_markdown: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    topic: str = "Unknown"
    keywords: List[str] = Field(default_factory=list)
    summary: str = ""
    content_type: str = "notes"
    has_diagrams: bool = False
    has_equations: bool = False
    has_tables: bool = False
    ink_quality: str = "unknown"
    writing_style: str = "unknown"
    language: str = "en"
    correction_notes: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CourseMetadata:
    """User-supplied metadata attached to every chunk from this course."""
    course_id:   str
    course_name: str
    subject:     str       = ""
    instructor:  str       = ""
    semester:    str       = ""
    university:  str       = ""
    tags:        List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "course_id":   self.course_id,
            "course_name": self.course_name,
            "subject":     self.subject,
            "instructor":  self.instructor,
            "semester":    self.semester,
            "university":  self.university,
            "tags":        self.tags,
        }


@dataclass
class PageOCRResult:
    """OCR output for one page, returned by Gemini."""
    page_number:        int
    transcript:         str          # verbatim text
    structured_markdown: str         # markdown re-rendering
    confidence:         float        # 0.0 – 1.0
    topic:              str
    keywords:           List[str]
    summary:            str          # 2-3 sentences
    content_type:       str          # notes | diagram | equation | table | mixed
    has_diagrams:       bool
    has_equations:      bool
    has_tables:         bool
    ink_quality:        str          # good | faded | smudged | unknown
    writing_style:      str          # print | cursive | mixed | unknown
    language:           str          # ISO-639-1
    correction_notes:   str          # uncertain words logged by the model
    model_used:         str          # which Gemini model produced this

    def is_low_confidence(self, threshold: float) -> bool:
        return self.confidence < threshold


@dataclass
class Chunk:
    """One vector unit to be stored in Pinecone."""
    chunk_id:    str
    text:        str       # context-header + chunk body
    page_number: int
    chunk_index: int
    source_file: str
    course:      CourseMetadata
    ocr:         PageOCRResult

    def metadata(self) -> dict:
        """Flat dict stored as Pinecone vector metadata."""
        return {
            **self.course.to_dict(),
            "source_file":    self.source_file,
            "page_number":    self.page_number,
            "chunk_index":    self.chunk_index,
            "topic":          self.ocr.topic,
            "content_type":   self.ocr.content_type,
            "keywords":       self.ocr.keywords,
            "summary":        self.ocr.summary,
            "confidence":     round(self.ocr.confidence, 4),
            "model_used":     self.ocr.model_used,
            "has_diagrams":   self.ocr.has_diagrams,
            "has_equations":  self.ocr.has_equations,
            "has_tables":     self.ocr.has_tables,
            "ink_quality":    self.ocr.ink_quality,
            "writing_style":  self.ocr.writing_style,
            "language":       self.ocr.language,
            "correction_notes": self.ocr.correction_notes,
            "text_preview":   self.text[:300],
        }


@dataclass
class PipelineResult:
    """Returned by run_pipeline(); contains full audit trail."""
    source_file:    str
    course:         CourseMetadata
    ocr_results:    List[PageOCRResult] = field(default_factory=list)
    chunks:         List[Chunk]         = field(default_factory=list)
    upserted_count: int                 = 0
    flash_pages:    int                 = 0
    pro_pages:      int                 = 0
    errors:         List[str]           = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "─" * 58,
            f"  Pipeline Result — {self.source_file}",
            "─" * 58,
            f"  Course    : {self.course.course_id} — {self.course.course_name}",
            f"  Pages     : {len(self.ocr_results)}  "
            f"(Flash: {self.flash_pages}  Pro: {self.pro_pages})",
            f"  Chunks    : {len(self.chunks)}",
            f"  Upserted  : {self.upserted_count} vectors → Pinecone",
        ]
        if self.errors:
            lines.append(f"  Errors    : {len(self.errors)}")
            for e in self.errors:
                lines.append(f"    • {e}")
        lines.append("─" * 58)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

OCR_SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8") if PROMPT_FILE.exists() else """\
You are an expert OCR and document-analysis model specialised in handwritten academic notes.
Your task is to analyse an image of a handwritten page and return a single JSON object.

STRICT RULES
============
1. Return ONLY valid JSON — no markdown fences, no prose, no comments.
2. Every field listed in the schema must be present.
3. All string values must be properly escaped (no raw newlines inside strings; use \\n).
4. "confidence" must reflect your TRUE certainty in the transcription quality:
   • 0.9–1.0 → clean, clear print; near-certain transcription
   • 0.75–0.89 → mostly legible; a few uncertain words
   • 0.5–0.74 → partially legible; cursive or faded sections
   • 0.0–0.49 → heavily degraded; many uncertain passages

TRANSCRIPTION GUIDELINES
=========================
• Preserve ALL visible text, including margin notes, arrows, labels, and corrections.
• Maintain original structure: headings, sub-headings, bullet points, numbered lists.
• For equations/formulas: use LaTeX notation inside $...$ for inline, $$...$$ for block.
• For diagrams: describe the diagram briefly inside [DIAGRAM: ...] tags, then transcribe
  all labels / annotations verbatim.
• For tables: reproduce as a Markdown table in "structured_markdown".
• If a word is ambiguous, write your best guess and log it in "correction_notes".
• Do NOT invent or paraphrase — transcribe exactly what is written.

JSON SCHEMA (return exactly this structure)
============================================
{
  "transcript": "<full verbatim text of the entire page>",
  "structured_markdown": "<markdown version with headings, bullets, LaTeX, tables>",
  "confidence": <float 0.0–1.0>,
  "topic": "<main topic or section heading of this page>",
  "keywords": ["<term1>", "<term2>", "..."],
  "summary": "<2-3 sentence summary of the page content>",
  "content_type": "<notes|diagram|equation|table|mixed>",
  "has_diagrams": <true|false>,
  "has_equations": <true|false>,
  "has_tables": <true|false>,
  "ink_quality": "<good|faded|smudged|unknown>",
  "writing_style": "<print|cursive|mixed|unknown>",
  "language": "<ISO-639-1 code>",
  "correction_notes": "<comma-separated uncertain words and guesses, or empty string>"
}
"""

OCR_USER_PROMPT = """\
Analyse the handwritten page shown in the image.
Return a single JSON object following the schema exactly.
Do not include any text outside the JSON object.
"""

PRO_ESCALATION_USER_PROMPT = """\
A previous OCR attempt on this handwritten page returned LOW CONFIDENCE ({confidence:.2f}).

Previous transcript attempt:
---
{previous_transcript}
---

Uncertain words flagged by previous model:
{correction_notes}

Please re-analyse the SAME image with maximum care.
Pay special attention to:
  • Words or passages flagged as uncertain above
  • Any smudged, faded, or densely-written sections
  • Symbols, superscripts, subscripts, and mathematical notation

Return a SINGLE corrected JSON object following the same schema.
Do not include any text outside the JSON object.
"""

EMBEDDING_TASK_TYPE       = "RETRIEVAL_DOCUMENT"
EMBEDDING_QUERY_TASK_TYPE = "RETRIEVAL_QUERY"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INGESTOR  (PDF / image → list of PIL images)
# ══════════════════════════════════════════════════════════════════════════════

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _resize_if_needed(img) -> Any:
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= cfg.max_image_px:
        return img
    scale = cfg.max_image_px / long_edge
    from PIL import Image
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _load_pdf(path: Path) -> List[Tuple[int, Any]]:
    log.info("Rasterising PDF '%s' at %d DPI ...", path.name, cfg.pdf_dpi)

    try:
        import fitz
        from PIL import Image

        doc = fitz.open(str(path))
        pages = []
        matrix = fitz.Matrix(cfg.pdf_dpi / 72, cfg.pdf_dpi / 72)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append((i, _resize_if_needed(img)))
        doc.close()
        return pages
    except ImportError:
        log.warning("PyMuPDF is not installed. Falling back to pdf2image.")
    except Exception as exc:
        log.warning("PyMuPDF failed for '%s': %s. Falling back to pdf2image.", path.name, exc)

    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF or pdf2image is required for handwritten PDF processing.\n"
            "Install PyMuPDF with: pip install PyMuPDF"
        ) from exc

    try:
        pages = convert_from_path(str(path), dpi=cfg.pdf_dpi)
        return [(i + 1, _resize_if_needed(p.convert("RGB"))) for i, p in enumerate(pages)]
    except Exception as exc:
        raise RuntimeError(f"Could not rasterise PDF '{path.name}' with PyMuPDF or pdf2image: {exc}") from exc


def _load_image(path: Path) -> List[Tuple[int, Any]]:
    from PIL import Image
    log.info("Loading image '%s' …", path.name)
    return [(1, _resize_if_needed(Image.open(path).convert("RGB")))]


def ingest(filepath: str) -> List[Tuple[int, Any]]:
    """
    Load a PDF or image file.
    Returns [(page_number, PIL.Image), ...] — always 1-based page numbers.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in _IMAGE_EXTENSIONS:
        return _load_image(path)
    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        f"Supported: .pdf, {', '.join(sorted(_IMAGE_EXTENSIONS))}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OCR ENGINE  (Gemini Flash → Pro escalation)
# ══════════════════════════════════════════════════════════════════════════════

def _gemini_client():
    try:
        from google import genai
        return genai.Client(api_key=cfg.gemini_api_key)
    except ImportError as exc:
        raise ImportError("pip install google-genai") from exc


def _pil_to_base64(img, fmt: str = "JPEG") -> Tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode(), f"image/{fmt.lower()}"


def _extract_json(raw: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from model response:\n{raw[:500]}")


def _dict_to_ocr(data: dict, page_number: int, model_used: str) -> PageOCRResult:
    parsed = HandwrittenOCRResponse.model_validate(data)
    return PageOCRResult(
        page_number=page_number,
        transcript=parsed.transcript,
        structured_markdown=parsed.structured_markdown,
        confidence=parsed.confidence,
        topic=parsed.topic,
        keywords=parsed.keywords,
        summary=parsed.summary,
        content_type=parsed.content_type,
        has_diagrams=parsed.has_diagrams,
        has_equations=parsed.has_equations,
        has_tables=parsed.has_tables,
        ink_quality=parsed.ink_quality,
        writing_style=parsed.writing_style,
        language=parsed.language,
        correction_notes=parsed.correction_notes,
        model_used=model_used,
    )


def _call_gemini_ocr(client, model_name: str, img_b64: str, img_mime: str, user_text: str) -> str:
    from google.genai import types
    contents = [
        types.Content(role="user", parts=[
            types.Part(inline_data=types.Blob(mime_type=img_mime, data=img_b64)),
            types.Part(text=user_text),
        ])
    ]
    config = types.GenerateContentConfig(
        system_instruction=OCR_SYSTEM_PROMPT,
        temperature=0.1,
        max_output_tokens=8192,
    )
    return client.models.generate_content(model=model_name, contents=contents, config=config).text


def ocr_page(img, page_number: int) -> PageOCRResult:
    """
    OCR one page image.
    Tries Gemini Flash first; escalates to Pro if confidence < threshold.
    Returns the best PageOCRResult with model_used set.
    """
    client = _gemini_client()
    img_b64, img_mime = _pil_to_base64(img)
    flash_result: Optional[PageOCRResult] = None

    # ── Flash attempt ─────────────────────────────────────────────────────────
    log.info("Page %d: OCR with Flash …", page_number)
    try:
        raw = _call_gemini_ocr(client, cfg.flash_model, img_b64, img_mime, OCR_USER_PROMPT)
        flash_result = _dict_to_ocr(_extract_json(raw), page_number, cfg.flash_model)
    except Exception as exc:
        log.warning("Page %d: Flash failed (%s) — escalating to Pro.", page_number, exc)

    # ── Confidence check ──────────────────────────────────────────────────────
    if flash_result and not flash_result.is_low_confidence(cfg.confidence_threshold):
        log.info("Page %d: Flash conf=%.2f ≥ %.2f — keeping Flash result.",
                 page_number, flash_result.confidence, cfg.confidence_threshold)
        return flash_result

    # ── Pro escalation ────────────────────────────────────────────────────────
    if flash_result:
        log.info("Page %d: Flash conf=%.2f < %.2f — escalating to Pro.",
                 page_number, flash_result.confidence, cfg.confidence_threshold)
        pro_prompt = PRO_ESCALATION_USER_PROMPT.format(
            confidence=flash_result.confidence,
            previous_transcript=flash_result.transcript[:2000],
            correction_notes=flash_result.correction_notes or "(none flagged)",
        )
    else:
        log.info("Page %d: Flash failed — sending directly to Pro.", page_number)
        pro_prompt = OCR_USER_PROMPT

    try:
        raw = _call_gemini_ocr(client, cfg.pro_model, img_b64, img_mime, pro_prompt)
        pro_result = _dict_to_ocr(_extract_json(raw), page_number, cfg.pro_model)
        log.info("Page %d: Pro conf=%.2f.", page_number, pro_result.confidence)
        return pro_result
    except Exception as exc:
        log.error("Page %d: Pro also failed (%s).", page_number, exc)
        if flash_result:
            log.warning("Page %d: Returning partial Flash result.", page_number)
            return flash_result
        raise RuntimeError(f"OCR failed on page {page_number}: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CHUNKER  (semantic split → merge tiny → split big)
# ══════════════════════════════════════════════════════════════════════════════

_HEADING_SPLIT = re.compile(r"(?=^#{1,3}\s)", re.MULTILINE)
_PARA_SPLIT    = re.compile(r"\n{2,}")
_SENTENCE_END  = re.compile(r"(?<=[.!?])\s+")


def _raw_segments(markdown: str) -> List[str]:
    parts: List[str] = []
    for heading_part in _HEADING_SPLIT.split(markdown):
        for para in _PARA_SPLIT.split(heading_part.strip()):
            if para.strip():
                parts.append(para.strip())
    return parts


def _merge_tiny(segs: List[str], min_chars: int) -> List[str]:
    out: List[str] = []
    for s in segs:
        if out and len(s) < min_chars:
            out[-1] += "\n\n" + s
        else:
            out.append(s)
    return out


def _split_big(segs: List[str], max_chars: int) -> List[str]:
    out: List[str] = []
    for s in segs:
        if len(s) <= max_chars:
            out.append(s)
            continue
        current = ""
        for sentence in _SENTENCE_END.split(s):
            if len(current) + len(sentence) + 1 > max_chars and current:
                out.append(current.strip())
                current = sentence
            else:
                current = (current + " " + sentence).strip() if current else sentence
        if current:
            out.append(current.strip())
    return out


def _context_header(course: CourseMetadata, ocr: PageOCRResult) -> str:
    parts = [p for p in [course.course_id, course.course_name, ocr.topic,
                          f"Page {ocr.page_number}"] if p]
    return "[" + " | ".join(parts) + "]"


def chunk_page(ocr: PageOCRResult, course: CourseMetadata, source_file: str) -> List[Chunk]:
    """Split one PageOCRResult into Chunk objects."""
    text = (ocr.structured_markdown or ocr.transcript).strip()
    if not text:
        return []
    segs = _split_big(_merge_tiny(_raw_segments(text), cfg.min_chunk_chars), cfg.max_chunk_chars)
    if not segs:
        return []
    header   = _context_header(course, ocr)
    safe_src = Path(source_file).stem.replace(" ", "_")
    return [
        Chunk(
            chunk_id    = f"{safe_src}::p{ocr.page_number}::c{i}",
            text        = f"{header}\n\n{seg}",
            page_number = ocr.page_number,
            chunk_index = i,
            source_file = source_file,
            course      = course,
            ocr         = ocr,
        )
        for i, seg in enumerate(segs)
    ]


def chunk_all_pages(ocr_results: List[PageOCRResult],
                    course: CourseMetadata,
                    source_file: str) -> List[Chunk]:
    """Chunk every page and return a flat list."""
    all_chunks: List[Chunk] = []
    for ocr in ocr_results:
        all_chunks.extend(chunk_page(ocr, course, source_file))
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EMBEDDER  (Gemini embedding, batched)
# ══════════════════════════════════════════════════════════════════════════════

def _embed_batch(client, texts: List[str], task_type: str) -> List[List[float]]:
    from google.genai import types
    response = client.models.embed_content(
        model=cfg.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=cfg.pinecone_dimension,
        ),
    )
    return [emb.values for emb in response.embeddings]


def embed_chunks(chunks: List[Chunk]) -> List[List[float]]:
    """Embed all chunks in batches. Returns vectors in the same order."""
    if not chunks:
        return []
    client = _gemini_client()
    texts  = [c.text for c in chunks]
    all_vecs: List[List[float]] = []
    for start in range(0, len(texts), cfg.embed_batch_size):
        batch = texts[start:start + cfg.embed_batch_size]
        log.info("Embedding chunks %d–%d of %d …", start + 1,
                 min(start + cfg.embed_batch_size, len(texts)), len(texts))
        all_vecs.extend(_embed_batch(client, batch, EMBEDDING_TASK_TYPE))
    log.info("Embedded %d chunks → %d-dim vectors.", len(chunks),
             len(all_vecs[0]) if all_vecs else 0)
    return all_vecs


def embed_query(query_text: str) -> List[float]:
    """Embed a single search query (RETRIEVAL_QUERY task type)."""
    from google.genai import types
    client   = _gemini_client()
    response = client.models.embed_content(
        model=cfg.embedding_model,
        contents=[query_text],
        config=types.EmbedContentConfig(
            task_type=EMBEDDING_QUERY_TASK_TYPE,
            output_dimensionality=cfg.pinecone_dimension,
        ),
    )
    return response.embeddings[0].values


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — VECTOR STORE  (Pinecone upsert / query / stats)
# ══════════════════════════════════════════════════════════════════════════════

def _pinecone_client():
    try:
        from pinecone import Pinecone
        return Pinecone(api_key=cfg.pinecone_api_key)
    except ImportError as exc:
        raise ImportError("pip install pinecone") from exc


def get_or_create_index():
    """Return the Pinecone Index, creating it automatically if needed."""
    from pinecone import ServerlessSpec
    pc         = _pinecone_client()
    idx_name   = cfg.pinecone_index_name
    existing   = [i.name for i in pc.list_indexes()]
    if idx_name not in existing:
        log.info("Creating Pinecone index '%s' (dim=%d) …", idx_name, cfg.pinecone_dimension)
        pc.create_index(
            name      = idx_name,
            dimension = cfg.pinecone_dimension,
            metric    = cfg.pinecone_metric,
            spec      = ServerlessSpec(cloud=cfg.pinecone_cloud, region=cfg.pinecone_region),
        )
        for _ in range(30):
            if pc.describe_index(idx_name).status.get("ready", False):
                break
            time.sleep(2)
        log.info("Index '%s' ready.", idx_name)
    return pc.Index(idx_name)


def upsert_chunks(chunks: List[Chunk], vectors: List[List[float]]) -> int:
    """
    Upsert chunk vectors into Pinecone.
    Returns the number of vectors successfully upserted.
    """
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch.")
    index     = get_or_create_index()
    namespace = cfg.pinecone_namespace
    records   = [{"id": c.chunk_id, "values": v, "metadata": c.metadata()}
                 for c, v in zip(chunks, vectors)]
    upserted  = 0
    for start in range(0, len(records), 100):
        batch = records[start:start + 100]
        log.info("Upserting vectors %d–%d …", start + 1, start + len(batch))
        upserted += index.upsert(vectors=batch, namespace=namespace).upserted_count
    log.info("Upserted %d vectors total.", upserted)
    return upserted


def query_notes(
    query_vector: List[float],
    course_id: Optional[str] = None,
    top_k: int = 5,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone for the closest chunks to query_vector.

    Args:
        query_vector : From embed_query().
        course_id    : Optional — narrows search to one course.
        top_k        : Number of results.
        filter_dict  : Additional Pinecone metadata filters.

    Returns:
        List of {id, score, metadata} dicts.
    """
    index = get_or_create_index()
    pf: Dict[str, Any] = {}
    if course_id:
        pf["course_id"] = {"$eq": course_id}
    if filter_dict:
        pf.update(filter_dict)
    kwargs: Dict[str, Any] = dict(
        vector=query_vector,
        top_k=top_k,
        namespace=cfg.pinecone_namespace,
        include_metadata=True,
    )
    if pf:
        kwargs["filter"] = pf
    resp = index.query(**kwargs)
    return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in resp.matches]


def index_stats() -> Dict[str, Any]:
    """Return Pinecone index statistics."""
    index = get_or_create_index()
    stats = index.describe_index_stats()
    return {
        "index_name":         cfg.pinecone_index_name,
        "total_vector_count": stats.total_vector_count,
        "dimension":          stats.dimension,
        "namespaces": {
            ns: info.vector_count
            for ns, info in (stats.namespaces or {}).items()
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}m"


def run_pipeline(
    filepath: str,
    course_id: str,
    course_name: str,
    subject: str = "",
    instructor: str = "",
    semester: str = "",
    university: str = "",
    tags: Optional[List[str]] = None,
) -> PipelineResult:
    """
    Run the full 5-stage handwritten notes ingestion pipeline.

    Args:
        filepath    : Path to PDF or image file.
        course_id   : Unique course identifier (Pinecone filter key).
        course_name : Human-readable course name.
        subject     : Academic subject.
        instructor  : Instructor name.
        semester    : Semester / term string.
        university  : University or institution name.
        tags        : User-supplied topic tags.

    Returns:
        PipelineResult with full audit trail.
    """
    cfg.validate()

    course = CourseMetadata(
        course_id=course_id, course_name=course_name,
        subject=subject, instructor=instructor,
        semester=semester, university=university,
        tags=tags or [],
    )
    result = PipelineResult(source_file=Path(filepath).name, course=course)
    t_total = time.time()

    log.info("=" * 60)
    log.info("Pipeline start: %s  →  course=%s", result.source_file, course_id)
    log.info("=" * 60)

    # 1 — Ingest
    t = time.time()
    log.info("[1/5] Ingest …")
    try:
        pages = ingest(filepath)
        log.info("      %d page(s) loaded in %s.", len(pages), _elapsed(t))
    except Exception as exc:
        log.error("[1/5] Ingest failed: %s", exc)
        result.errors.append(f"Ingest failed: {exc}")
        return result

    # 2 — OCR
    t = time.time()
    log.info("[2/5] OCR (%d page(s)) …", len(pages))
    for page_num, img in pages:
        try:
            ocr = ocr_page(img, page_num)
            result.ocr_results.append(ocr)
            if cfg.flash_model in ocr.model_used:
                result.flash_pages += 1
            else:
                result.pro_pages += 1
            log.info("      Page %d: conf=%.2f  model=%s  topic=%s",
                     page_num, ocr.confidence,
                     "Flash" if cfg.flash_model in ocr.model_used else "Pro",
                     ocr.topic[:50])
        except Exception as exc:
            msg = f"OCR error on page {page_num}: {exc}"
            log.error("      %s", msg)
            result.errors.append(msg)

    log.info("      OCR done in %s — Flash=%d  Pro=%d  Errors=%d",
             _elapsed(t), result.flash_pages, result.pro_pages, len(result.errors))

    if not result.ocr_results:
        result.errors.append("All pages failed OCR.")
        return result

    # 3 — Chunk
    t = time.time()
    log.info("[3/5] Chunking …")
    result.chunks = chunk_all_pages(result.ocr_results, course, result.source_file)
    log.info("      %d chunks produced in %s.", len(result.chunks), _elapsed(t))

    if not result.chunks:
        log.warning("No chunks produced — pages may have been blank.")
        return result

    # 4 — Embed
    t = time.time()
    log.info("[4/5] Embedding %d chunks …", len(result.chunks))
    try:
        vectors = embed_chunks(result.chunks)
        log.info("      Embedding done in %s.", _elapsed(t))
    except Exception as exc:
        log.error("[4/5] Embedding failed: %s", exc)
        result.errors.append(f"Embedding failed: {exc}")
        return result

    # 5 — Upsert
    t = time.time()
    log.info("[5/5] Upserting to Pinecone …")
    try:
        result.upserted_count = upsert_chunks(result.chunks, vectors)
        log.info("      Upserted %d vectors in %s.", result.upserted_count, _elapsed(t))
    except Exception as exc:
        log.error("[5/5] Upsert failed: %s", exc)
        result.errors.append(f"Upsert failed: {exc}")

    log.info("=" * 60)
    log.info("Pipeline finished in %s.", _elapsed(t_total))
    log.info("=" * 60)
    print(result.summary())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def _cli_ingest(args: argparse.Namespace) -> None:
    run_pipeline(
        filepath=args.filepath,
        course_id=args.course_id,
        course_name=args.course_name or "",
        subject=args.subject or "",
        instructor=args.instructor or "",
        semester=args.semester or "",
        university=args.university or "",
        tags=args.tags or [],
    )


def _cli_query(args: argparse.Namespace) -> None:
    print(f"\nSearching: '{args.query}' …\n")
    vec     = embed_query(args.query)
    results = query_notes(vec, course_id=args.course_id or None, top_k=args.top_k)
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"[{i}] Score: {r['score']:.4f}")
        print(f"     Course : {m.get('course_id')} — {m.get('course_name')}")
        print(f"     Page   : {m.get('page_number')}  |  Topic: {m.get('topic')}")
        print(f"     Model  : {m.get('model_used')}  |  Conf: {m.get('confidence')}")
        print(f"     Preview: {m.get('text_preview', '')[:200]}")
        print()


def _cli_stats(_args: argparse.Namespace) -> None:
    print(json.dumps(index_stats(), indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="handwritten_pipeline.py",
        description="Handwritten Notes Ingestion Pipeline — Gemini + Pinecone",
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    ing = sub.add_parser("ingest", help="Ingest a PDF or image file.")
    ing.add_argument("filepath")
    _ingest_flags(ing)

    # query
    q = sub.add_parser("query", help="Semantic search over ingested notes.")
    q.add_argument("query")
    q.add_argument("--course-id")
    q.add_argument("--top-k", type=int, default=5)

    # stats
    sub.add_parser("stats", help="Show Pinecone index statistics.")

    # positional fallback (no subcommand)
    parser.add_argument("filepath", nargs="?")
    _ingest_flags(parser)
    return parser


def _ingest_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--course-id",   required=False)
    p.add_argument("--course-name", default="")
    p.add_argument("--subject",     default="")
    p.add_argument("--instructor",  default="")
    p.add_argument("--semester",    default="")
    p.add_argument("--university",  default="")
    p.add_argument("--tags", nargs="*", default=[])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "query":
        _cli_query(args)
    elif args.command == "stats":
        _cli_stats(args)
    elif args.command == "ingest":
        if not args.filepath:
            parser.error("filepath is required for ingest.")
        _cli_ingest(args)
    elif args.filepath:
        if not args.course_id:
            parser.error("--course-id is required.")
        _cli_ingest(args)
    else:
        parser.print_help()
        sys.exit(1)

