"""
handwritten_pipeline.py  (v2 - redesigned)
==========================================
Redesigned handwritten notes ingestion pipeline.

Architecture (v2)
-----------------
1. Ingest     - PDF/image -> rasterised PIL images (1-based page numbers)
2. OCR        - ALL pages concurrently via asyncio.gather() + semaphore;
                 each page: Gemini Flash first; escalate to Pro if confidence low.
                 Returns typed PageElement[] (Title, Heading, Paragraph, ...).
3. Pool       - All PageElements from all pages merged in page-number order into RAM.
4. Chunk      - Global cross-page semantic chunker:
                  * heading-anchored (anchor_title / anchor_heading carried across pages)
                  * Diagram / Table always isolated in their own standalone chunk
                  * 150-char overlap on hard-cap splits; NO overlap on heading splits
                  * Tiny chunk merging (never across Title/Heading boundary)
5. Embed      - Async concurrent: semaphore-bounded asyncio.to_thread() calls,
                  processed in groups of EMBED_GROUP_SIZE * EMBED_BATCH_SIZE per gather()
6. Upsert     - Pinecone batched at 100 vectors / call

OCR consistency guarantee
-------------------------
Each page OCR is fully independent (Gemini sees only that page image).
After asyncio.gather(), results are sorted by page_number before building
the element pool, guaranteeing deterministic sequential order regardless of
which pages finish OCR first.

Quick start
-----------
    pip install google-genai pinecone Pillow pdf2image python-dotenv

    python ingestion_pipeline.py notes.pdf --course-id CS101 --course-name "Intro to CS"

Env vars (.env):
    GEMINI_API_KEY=...
    PINECONE_API_KEY=...
    CONFIDENCE_THRESHOLD=0.75      # OCR escalation threshold
    OCR_CONCURRENCY=5              # max concurrent OCR page calls
    EMBED_CONCURRENCY=10           # max concurrent embed calls (semaphore)
    EMBED_BATCH_SIZE=10            # chunks per embed task group
    EMBED_GROUP_SIZE=10            # groups per asyncio.gather() call
"""

from __future__ import annotations

# -- stdlib --------------------------------------------------------------------
import argparse
import asyncio
import base64
import hashlib
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

# ==============================================================================
# SECTION 1 - CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    # API keys
    gemini_api_key:   str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
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
    pinecone_cloud:  str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1"))
    pinecone_metric: str = "cosine"

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

    # OCR escalation + concurrency
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    )
    ocr_concurrency: int = field(
        default_factory=lambda: int(os.getenv("OCR_CONCURRENCY", "5"))
    )

    # Ingestor
    pdf_dpi:      int = 150
    max_image_px: int = 2048   # long-edge cap before sending to Gemini

    # Chunker
    # soft_target_chars: preferred chunk size — split at semantic boundary when reached
    # max_chunk_chars:   hard maximum — forced split if exceeded (default 2500)
    # overlap_chars:     chars carried from previous chunk ONLY on hard/forced splits
    soft_target_chars: int = field(default_factory=lambda: int(os.getenv("SOFT_TARGET_CHARS", "1500")))
    min_chunk_chars:   int = 80
    max_chunk_chars:   int = field(default_factory=lambda: int(os.getenv("MAX_CHUNK_CHARS",   "2500")))
    overlap_chars:     int = field(default_factory=lambda: int(os.getenv("OVERLAP_CHARS",     "120")))


    # Embedder
    embed_batch_size:  int = field(default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE",  "10")))
    embed_group_size:  int = field(default_factory=lambda: int(os.getenv("EMBED_GROUP_SIZE",  "10")))
    embed_concurrency: int = field(default_factory=lambda: int(os.getenv("EMBED_CONCURRENCY", "10")))

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


# ==============================================================================
# SECTION 2 - DATA MODELS
# ==============================================================================

ELEMENT_TYPES = frozenset({
    "Title", "Heading", "Paragraph", "ListBlock",
    "Equation", "Diagram", "Table", "Caption", "MarginNote",
})


@dataclass
class PageElement:
    """One structural element extracted from a handwritten page by the OCR model."""
    element_type:  str    # Title | Heading | Paragraph | ListBlock | Equation
                          # | Diagram | Table | Caption | MarginNote
    content:       str    # verbatim text; LaTeX for equations; markdown for tables
    page_number:   int
    position_hint: str    # top | middle | bottom
    confidence:    float  # per-element OCR confidence

    def __post_init__(self) -> None:
        if self.element_type not in ELEMENT_TYPES:
            self.element_type = "Paragraph"
        self.position_hint = self.position_hint.lower()
        if self.position_hint not in {"top", "middle", "bottom"}:
            self.position_hint = "middle"


class HandwrittenOCRResponse(BaseModel):
    transcript:          str = ""
    structured_markdown: str = ""
    confidence:          float = Field(0.5, ge=0.0, le=1.0)
    topic:               str = "Unknown"
    keywords:            List[str] = Field(default_factory=list)
    summary:             str = ""
    content_type:        str = "notes"
    has_diagrams:        bool = False
    has_equations:       bool = False
    has_tables:          bool = False
    ink_quality:         str = "unknown"
    writing_style:       str = "unknown"
    language:            str = "en"
    correction_notes:    str = ""
    elements:            List[Dict[str, Any]] = Field(default_factory=list)


@dataclass
class CourseMetadata:
    """User-supplied metadata attached to every chunk."""
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
    """OCR output for one page."""
    page_number:         int
    elements:            List[PageElement]
    transcript:          str
    structured_markdown: str
    confidence:          float
    topic:               str
    keywords:            List[str]
    summary:             str
    content_type:        str
    has_diagrams:        bool
    has_equations:       bool
    has_tables:          bool
    ink_quality:         str
    writing_style:       str
    language:            str
    correction_notes:    str
    model_used:          str

    def is_low_confidence(self, threshold: float) -> bool:
        return self.confidence < threshold


@dataclass
class Chunk:
    """One vector unit for Pinecone - v2 with cross-page metadata."""
    chunk_id:       str
    text:           str
    page_start:     int
    page_end:       int
    chunk_index:    int
    element_types:  List[str]
    anchor_title:   str
    anchor_heading: str
    has_diagram:    bool
    has_equation:   bool
    has_table:      bool
    source_file:    str
    course:         CourseMetadata
    ocr_model_used: str
    avg_confidence: float

    def metadata(self) -> dict:
        return {
            **self.course.to_dict(),
            "source_file":    self.source_file,
            "page_start":     self.page_start,
            "page_end":       self.page_end,
            "chunk_index":    self.chunk_index,
            "element_types":  ", ".join(self.element_types),
            "anchor_title":   self.anchor_title,
            "anchor_heading": self.anchor_heading,
            "has_diagram":    self.has_diagram,
            "has_equation":   self.has_equation,
            "has_table":      self.has_table,
            "ocr_model_used": self.ocr_model_used,
            "avg_confidence": round(self.avg_confidence, 4),
            "text_preview":   self.text[:300],
        }


@dataclass
class PipelineResult:
    """Returned by run_pipeline() - full audit trail."""
    source_file:    str
    course:         CourseMetadata
    ocr_results:    List[PageOCRResult] = field(default_factory=list)
    element_pool:   List[PageElement]   = field(default_factory=list)
    chunks:         List[Chunk]         = field(default_factory=list)
    upserted_count: int                 = 0
    flash_pages:    int                 = 0
    pro_pages:      int                 = 0
    errors:         List[str]           = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "-" * 62,
            f"  Pipeline v2 Result: {self.source_file}",
            "-" * 62,
            f"  Course    : {self.course.course_id} - {self.course.course_name}",
            f"  Pages     : {len(self.ocr_results)}  (Flash:{self.flash_pages}  Pro:{self.pro_pages})",
            f"  Elements  : {len(self.element_pool)}",
            f"  Chunks    : {len(self.chunks)}",
            f"  Upserted  : {self.upserted_count} vectors",
        ]
        if self.errors:
            lines.append(f"  Errors    : {len(self.errors)}")
            for e in self.errors:
                lines.append(f"    - {e}")
        lines.append("-" * 62)
        return "\n".join(lines)


# ==============================================================================
# SECTION 3 - OCR PROMPTS
# ==============================================================================

_OCR_PROMPT_FALLBACK = """\
You are an expert OCR and document-analysis model specialised in handwritten academic notes.
Analyse the handwritten page image and return a single JSON object.

STRICT RULES
============
1. Return ONLY valid JSON - no markdown fences, no prose, no comments.
2. All fields in the schema must be present.
3. All string values must be properly escaped.
4. confidence must reflect your TRUE certainty in transcription quality.

ELEMENT TYPES (use EXACTLY these strings)
=========================================
  Title      - top-level document or main section title
  Heading    - section/sub-section heading (H2/H3 level)
  Paragraph  - body text block
  ListBlock  - complete bulleted or numbered list
  Equation   - mathematical formula (LaTeX: $...$ inline, $$...$$ block)
  Diagram    - visual element; describe it then transcribe all labels
  Table      - tabular data as Markdown table
  Caption    - caption for a figure/diagram/table
  MarginNote - handwritten annotation in the margin

TRANSCRIPTION GUIDELINES
=========================
- Preserve ALL visible text including margin notes, arrows, labels, corrections.
- Equations: use LaTeX notation inside $...$ or $$$...$$$.
- Diagrams: write [DIAGRAM: description] then list all visible labels.
- Tables: render as Markdown table inside the Table element.
- Ambiguous word: write best guess and log in correction_notes.

JSON SCHEMA
===========
{
  "elements": [
    {
      "element_type": "<Title|Heading|Paragraph|ListBlock|Equation|Diagram|Table|Caption|MarginNote>",
      "content": "<verbatim text>",
      "position_hint": "<top|middle|bottom>",
      "confidence": <float 0.0-1.0>
    }
  ],
  "transcript": "<complete verbatim text of the entire page>",
  "structured_markdown": "<full markdown rendering>",
  "confidence": <overall page float 0.0-1.0>,
  "topic": "<main topic of this page>",
  "keywords": ["<term1>", "<term2>"],
  "summary": "<2-3 sentence summary>",
  "content_type": "<notes|diagram|equation|table|mixed>",
  "has_diagrams": <true|false>,
  "has_equations": <true|false>,
  "has_tables": <true|false>,
  "ink_quality": "<good|faded|smudged|unknown>",
  "writing_style": "<print|cursive|mixed|unknown>",
  "language": "<ISO-639-1 code>",
  "correction_notes": "<uncertain words, or empty string>"
}
"""

OCR_SYSTEM_PROMPT = (
    PROMPT_FILE.read_text(encoding="utf-8")
    if PROMPT_FILE.exists()
    else _OCR_PROMPT_FALLBACK
)

OCR_USER_PROMPT = (
    "Analyse the handwritten page shown in the image.\n"
    "Return a single JSON object following the schema exactly.\n"
    "Do not include any text outside the JSON object."
)

PRO_ESCALATION_USER_PROMPT = (
    "A previous OCR attempt returned LOW CONFIDENCE ({confidence:.2f}).\n\n"
    "Previous transcript:\n---\n{previous_transcript}\n---\n\n"
    "Uncertain words: {correction_notes}\n\n"
    "Re-analyse the SAME image with maximum care. "
    "Return a SINGLE corrected JSON object following the same schema."
)

EMBEDDING_TASK_TYPE       = "RETRIEVAL_DOCUMENT"
EMBEDDING_QUERY_TASK_TYPE = "RETRIEVAL_QUERY"


# ==============================================================================
# SECTION 4 - INGESTOR
# ==============================================================================

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
    log.info("Rasterising PDF %s at %d DPI ...", path.name, cfg.pdf_dpi)
    try:
        import fitz
        from PIL import Image
        doc    = fitz.open(str(path))
        pages  = []
        matrix = fitz.Matrix(cfg.pdf_dpi / 72, cfg.pdf_dpi / 72)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append((i, _resize_if_needed(img)))
        doc.close()
        return pages
    except ImportError:
        log.warning("PyMuPDF not installed - falling back to pdf2image.")
    except Exception as exc:
        log.warning("PyMuPDF failed (%s) - falling back to pdf2image.", exc)
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ImportError("Install PyMuPDF: pip install PyMuPDF") from exc
    try:
        pages = convert_from_path(str(path), dpi=cfg.pdf_dpi)
        return [(i + 1, _resize_if_needed(p.convert("RGB"))) for i, p in enumerate(pages)]
    except Exception as exc:
        raise RuntimeError(f"Could not rasterise PDF {path.name}: {exc}") from exc


def _load_image(path: Path) -> List[Tuple[int, Any]]:
    from PIL import Image
    return [(1, _resize_if_needed(Image.open(path).convert("RGB")))]


def ingest(filepath: str) -> List[Tuple[int, Any]]:
    """Load PDF or image. Returns [(page_number, PIL.Image), ...] 1-based."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in _IMAGE_EXTENSIONS:
        return _load_image(path)
    raise ValueError(f"Unsupported file type {suffix!r}.")


# ==============================================================================
# SECTION 5 - OCR ENGINE  (async Flash->Pro per page)
# ==============================================================================

def _gemini_client():
    from google import genai
    return genai.Client(api_key=cfg.gemini_api_key)


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
    raise ValueError(f"Could not parse JSON:\n{raw[:500]}")


def _parse_elements(data: dict, page_number: int) -> List[PageElement]:
    """Parse elements[] from OCR JSON; fall back to synthesising from transcript."""
    raw_elements = data.get("elements", [])
    if raw_elements:
        elements = []
        for e in raw_elements:
            if not isinstance(e, dict):
                continue
            content = str(e.get("content", "")).strip()
            if not content:
                continue
            elements.append(PageElement(
                element_type  = str(e.get("element_type", "Paragraph")),
                content       = content,
                page_number   = page_number,
                position_hint = str(e.get("position_hint", "middle")),
                confidence    = float(e.get("confidence", data.get("confidence", 0.5))),
            ))
        if elements:
            return elements

    # Fallback: synthesise from markdown/transcript
    log.warning("Page %d: no elements[] returned - synthesising from transcript.", page_number)
    text = (data.get("structured_markdown") or data.get("transcript") or "").strip()
    if not text:
        return []
    pg_conf = float(data.get("confidence", 0.5))
    result  = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if re.match(r"^#{1,3}\s", para):
            el_type = "Heading" if para.startswith("##") else "Title"
            para    = re.sub(r"^#+\s*", "", para).strip()
        elif re.match(r"^\$\$", para) or re.match(r"^\\\[", para):
            el_type = "Equation"
        elif re.match(r"^\|", para):
            el_type = "Table"
        elif re.match(r"^\[DIAGRAM:", para, re.IGNORECASE):
            el_type = "Diagram"
        else:
            el_type = "Paragraph"
        result.append(PageElement(
            element_type  = el_type,
            content       = para,
            page_number   = page_number,
            position_hint = "middle",
            confidence    = pg_conf,
        ))
    return result


def _dict_to_ocr(data: dict, page_number: int, model_used: str) -> PageOCRResult:
    parsed   = HandwrittenOCRResponse.model_validate(data)
    elements = _parse_elements(data, page_number)
    return PageOCRResult(
        page_number         = page_number,
        elements            = elements,
        transcript          = parsed.transcript,
        structured_markdown = parsed.structured_markdown,
        confidence          = parsed.confidence,
        topic               = parsed.topic,
        keywords            = parsed.keywords,
        summary             = parsed.summary,
        content_type        = parsed.content_type,
        has_diagrams        = parsed.has_diagrams,
        has_equations       = parsed.has_equations,
        has_tables          = parsed.has_tables,
        ink_quality         = parsed.ink_quality,
        writing_style       = parsed.writing_style,
        language            = parsed.language,
        correction_notes    = parsed.correction_notes,
        model_used          = model_used,
    )


def _call_gemini_ocr_sync(
    client, model_name: str, img_b64: str, img_mime: str, user_text: str,
) -> str:
    from google.genai import types
    contents = [
        types.Content(role="user", parts=[
            types.Part(inline_data=types.Blob(mime_type=img_mime, data=img_b64)),
            types.Part(text=user_text),
        ])
    ]
    config = types.GenerateContentConfig(
        system_instruction = OCR_SYSTEM_PROMPT,
        temperature        = 0.1,
        max_output_tokens  = 8192,
    )
    return client.models.generate_content(
        model=model_name, contents=contents, config=config
    ).text


def ocr_page(img, page_number: int) -> PageOCRResult:
    """Sync OCR for one page: Flash first, escalate to Pro if confidence low."""
    client            = _gemini_client()
    img_b64, img_mime = _pil_to_base64(img)
    flash_result: Optional[PageOCRResult] = None

    log.info("Page %d: Flash OCR ...", page_number)
    try:
        raw          = _call_gemini_ocr_sync(client, cfg.flash_model, img_b64, img_mime, OCR_USER_PROMPT)
        flash_result = _dict_to_ocr(_extract_json(raw), page_number, cfg.flash_model)
    except Exception as exc:
        log.warning("Page %d: Flash failed (%s) - escalating to Pro.", page_number, exc)

    if flash_result and not flash_result.is_low_confidence(cfg.confidence_threshold):
        log.info("Page %d: Flash conf=%.2f >= %.2f - keeping.",
                 page_number, flash_result.confidence, cfg.confidence_threshold)
        return flash_result

    pro_prompt = (
        PRO_ESCALATION_USER_PROMPT.format(
            confidence          = flash_result.confidence,
            previous_transcript = flash_result.transcript[:2000],
            correction_notes    = flash_result.correction_notes or "(none)",
        ) if flash_result else OCR_USER_PROMPT
    )
    log.info("Page %d: escalating to Pro ...", page_number)
    try:
        raw        = _call_gemini_ocr_sync(client, cfg.pro_model, img_b64, img_mime, pro_prompt)
        pro_result = _dict_to_ocr(_extract_json(raw), page_number, cfg.pro_model)
        log.info("Page %d: Pro conf=%.2f.", page_number, pro_result.confidence)
        return pro_result
    except Exception as exc:
        log.error("Page %d: Pro also failed (%s).", page_number, exc)
        if flash_result:
            return flash_result
        raise RuntimeError(f"OCR failed on page {page_number}: {exc}") from exc


async def _ocr_page_async(
    semaphore:   asyncio.Semaphore,
    img:         Any,
    page_number: int,
) -> Optional[PageOCRResult]:
    """
    Async wrapper around ocr_page().
    Runs the sync Gemini call in a thread pool via asyncio.to_thread().
    Bounded by semaphore to limit concurrent Gemini API calls.

    Consistency guarantee: each page is completely independent.
    After asyncio.gather(), results are sorted by page_number so the
    element pool is always assembled in correct sequential order.
    """
    async with semaphore:
        try:
            return await asyncio.to_thread(ocr_page, img, page_number)
        except Exception as exc:
            log.error("Page %d async OCR failed: %s", page_number, exc)
            return None


async def ocr_all_pages_async(
    pages: List[Tuple[int, Any]],
) -> Tuple[List[PageOCRResult], int, int, List[str]]:
    """
    OCR all pages concurrently, bounded by OCR_CONCURRENCY semaphore.

    No consistency issues: pages are independent. After gather(), results are
    sorted by page_number for deterministic element pool order.

    Returns: (ocr_results_sorted, flash_count, pro_count, error_list)
    """
    semaphore = asyncio.Semaphore(cfg.ocr_concurrency)
    log.info("OCR: %d pages, concurrency=%d", len(pages), cfg.ocr_concurrency)

    tasks = [
        _ocr_page_async(semaphore, img, page_num)
        for page_num, img in pages
    ]
    raw_results: List[Optional[PageOCRResult]] = await asyncio.gather(*tasks)

    ocr_results: List[PageOCRResult] = []
    flash_pages  = 0
    pro_pages    = 0
    errors: List[str] = []

    for page_num, raw in zip([p for p, _ in pages], raw_results):
        if raw is None:
            errors.append(f"OCR failed on page {page_num}")
            continue
        ocr_results.append(raw)
        if cfg.flash_model in raw.model_used:
            flash_pages += 1
        else:
            pro_pages += 1
        log.info("  Page %02d: conf=%.2f  model=%s  elements=%d  topic=%s",
                 raw.page_number, raw.confidence,
                 "Flash" if cfg.flash_model in raw.model_used else "Pro",
                 len(raw.elements), raw.topic[:50])

    # Sort by page_number - guarantees pool order regardless of completion order
    ocr_results.sort(key=lambda r: r.page_number)
    return ocr_results, flash_pages, pro_pages, errors


# ==============================================================================
# SECTION 6 - CROSS-PAGE CHUNKER  (refactored)
# ==============================================================================
#
# Size targets (configurable via Config, with sensible defaults):
#   SOFT_TARGET : 1200-1800 chars  - preferred chunk size; try to close here
#   HARD_MAX    : 2200-2800 chars  - never exceed; force split if reached
#   OVERLAP     : 80-150 chars     - carried into next chunk ONLY on hard splits;
#                                    never on natural semantic splits
#
# Split-boundary preference (best to worst):
#   1. Heading / Title boundary          (always split here)
#   2. Paragraph boundary after soft target
#   3. Sentence boundary (".", "!", "?")
#   4. Hard-cap forced split             (overlap applied here only)
# ==============================================================================

_STANDALONE_TYPES = frozenset({"Diagram", "Table"})
# Element types that are "anchor" type - they mark section boundaries
_ANCHOR_TYPES     = frozenset({"Title", "Heading"})
# Sentence-ending pattern used for fine-grained overlap trimming
_SENT_END_RE      = re.compile(r"(?<=[.!?])\s+")


def _chunk_id_for(source_file: str, index: int) -> str:
    raw = f"{Path(source_file).resolve()}::hw::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _context_header(
    course:         CourseMetadata,
    anchor_title:   str,
    anchor_heading: str,
    page_start:     int,
    page_end:       int,
) -> str:
    page_str = f"Page {page_start}" if page_start == page_end else f"Pages {page_start}-{page_end}"
    parts    = [p for p in [course.course_id, course.course_name,
                              anchor_title, anchor_heading, page_str] if p]
    return "[" + " | ".join(parts) + "]"


def _ocr_model_tag(elements: List[PageElement], ocr_results: List[PageOCRResult]) -> str:
    page_nums = {e.page_number for e in elements}
    models    = set()
    for r in ocr_results:
        if r.page_number in page_nums:
            models.add("flash" if cfg.flash_model in r.model_used else "pro")
    if not models:
        return "unknown"
    return "mixed" if len(models) > 1 else models.pop()


def _avg_confidence(elements: List[PageElement]) -> float:
    if not elements:
        return 0.0
    return round(sum(e.confidence for e in elements) / len(elements), 4)


def _element_to_text(el: PageElement) -> str:
    """Render a single PageElement to a display string."""
    if el.element_type == "Heading":
        return f"## {el.content}"
    if el.element_type == "Title":
        return f"# {el.content}"
    if el.element_type == "Diagram":
        return f"[DIAGRAM - Page {el.page_number}]\n{el.content}"
    if el.element_type == "Table":
        return f"[TABLE - Page {el.page_number}]\n{el.content}"
    if el.element_type == "MarginNote":
        return f"[MARGIN NOTE] {el.content}"
    return el.content  # Paragraph, ListBlock, Equation, Caption


def _elements_to_text(elements: List[PageElement]) -> str:
    """Concatenate a list of elements into a single text block."""
    return "\n\n".join(_element_to_text(el) for el in elements)


def _build_chunk(
    elements:       List[PageElement],
    chunk_index:    int,
    source_file:    str,
    course:         CourseMetadata,
    anchor_title:   str,
    anchor_heading: str,
    ocr_results:    List[PageOCRResult],
    overlap_prefix: str = "",   # overlap text carried from previous hard split
) -> Chunk:
    """Assemble a Chunk dataclass from a slice of PageElements."""
    page_nums = sorted({e.page_number for e in elements})
    body      = _elements_to_text(elements)

    # Prepend overlap prefix only when provided (hard-split case only)
    if overlap_prefix:
        body = overlap_prefix + "\n\n" + body

    el_types  = list(dict.fromkeys(e.element_type for e in elements))  # ordered unique
    header    = _context_header(
        course, anchor_title, anchor_heading,
        page_nums[0], page_nums[-1]
    )
    return Chunk(
        chunk_id       = _chunk_id_for(source_file, chunk_index),
        text           = f"{header}\n\n{body}",
        page_start     = page_nums[0],
        page_end       = page_nums[-1],
        chunk_index    = chunk_index,
        element_types  = el_types,
        anchor_title   = anchor_title,
        anchor_heading = anchor_heading,
        has_diagram    = any(e.element_type == "Diagram"  for e in elements),
        has_equation   = any(e.element_type == "Equation" for e in elements),
        has_table      = any(e.element_type == "Table"    for e in elements),
        source_file    = source_file,
        course         = course,
        ocr_model_used = _ocr_model_tag(elements, ocr_results),
        avg_confidence = _avg_confidence(elements),
    )


def _sentence_boundary_overlap(text: str, max_overlap: int) -> str:
    """
    Extract up to max_overlap characters from the END of `text`, trimmed to a
    complete sentence boundary whenever possible.

    Example: '...gravity. Kinetic energy is the energy of motion. An object...'
    With max_overlap=80 returns the last complete sentence that fits within 80 chars.
    This gives the next chunk a contextual opening instead of an arbitrary char slice.
    """
    if not text or max_overlap <= 0:
        return ""
    tail = text[-max_overlap * 2:]   # grab 2x window to find sentence boundary
    # Walk backwards through sentences to find the last full sentence that fits
    sentences = _SENT_END_RE.split(tail)
    overlap   = ""
    for sent in reversed(sentences):
        candidate = sent.strip()
        if candidate and len(candidate) <= max_overlap:
            overlap = candidate
            break
    # Fall back to raw character slice if no sentence fits
    return overlap if overlap else text[-max_overlap:].lstrip()


def _split_long_text(
    text:    str,
    hard_max: int,
    overlap:  int,
) -> List[Tuple[str, str]]:
    """
    Split a long string into segments that each fit within `hard_max` characters.
    Tries sentence boundaries first; falls back to hard char split.

    Returns List[(segment_text, overlap_for_next_chunk)].
    The overlap for the LAST segment is always "" (nothing follows it here).
    """
    if len(text) <= hard_max:
        return [(text, "")]

    results: List[Tuple[str, str]] = []
    remaining = text

    while len(remaining) > hard_max:
        # Try to split at a sentence boundary near hard_max
        window   = remaining[:hard_max]
        # Find the last sentence end inside the window
        last_dot = max(
            (m.end() for m in _SENT_END_RE.finditer(window)),
            default=0
        )
        if last_dot > hard_max // 2:
            # Good sentence split found well into the chunk
            segment   = remaining[:last_dot].rstrip()
            remaining = remaining[last_dot:].lstrip()
        else:
            # No sentence split found; hard character split
            segment   = remaining[:hard_max]
            remaining = remaining[max(0, hard_max - overlap):].lstrip()

        ovlp = _sentence_boundary_overlap(segment, overlap)
        results.append((segment, ovlp))

    if remaining:
        results.append((remaining, ""))

    return results


def chunk_element_pool(
    element_pool: List[PageElement],
    course:       CourseMetadata,
    source_file:  str,
    ocr_results:  List[PageOCRResult],
) -> List[Chunk]:
    """
    Advanced cross-page semantic chunker.

    Operates on the complete element pool (all pages, sorted by page_number).
    Produces heading-anchored chunks that may span page boundaries.

    Boundary priority
    -----------------
    1. Title / Heading   — always forces a chunk boundary; updates anchors
    2. Soft target hit   — look ahead for natural paragraph/sentence end
    3. Hard maximum hit  — forced split with sentence-boundary overlap carried forward
    4. Diagram / Table   — always emitted as an isolated standalone chunk

    Overlap policy
    --------------
    Overlap is applied ONLY on hard/forced splits (max_chunk_chars exceeded).
    When splitting at a natural boundary (Title, Heading, paragraph end after
    reaching soft target), NO overlap is added — context is already clean.
    """
    if not element_pool:
        return []

    # ── Size constants (from cfg — tune via env vars: SOFT_TARGET_CHARS, MAX_CHUNK_CHARS, OVERLAP_CHARS)
    SOFT_TARGET = cfg.soft_target_chars  # preferred chunk size (default 1500)
    HARD_MAX    = cfg.max_chunk_chars    # hard maximum (default 2500)
    OVERLAP     = cfg.overlap_chars      # overlap on hard splits only (default 120)


    # ── State variables ────────────────────────────────────────────────────────
    chunks:         List[Chunk]       = []
    current_els:    List[PageElement] = []
    current_chars:  int               = 0      # total chars in current_els (approx)
    anchor_title:   str               = ""
    anchor_heading: str               = ""
    chunk_index:    int               = 0
    pending_overlap: str              = ""     # overlap text from previous hard split

    # ── Helper: emit current_els as a Chunk ───────────────────────────────────
    def flush(use_overlap: bool = False) -> None:
        """
        Emit current_els as a chunk.
        use_overlap=True: prepend pending_overlap (hard-split continuation).
        use_overlap=False: no prefix (natural/anchor boundary).
        After flush, pending_overlap is always consumed and cleared.
        """
        nonlocal current_els, current_chars, chunk_index, pending_overlap
        if not current_els:
            pending_overlap = ""   # discard stale overlap if nothing to flush
            return
        prefix = pending_overlap if use_overlap else ""
        chunks.append(_build_chunk(
            elements       = current_els,
            chunk_index    = chunk_index,
            source_file    = source_file,
            course         = course,
            anchor_title   = anchor_title,
            anchor_heading = anchor_heading,
            ocr_results    = ocr_results,
            overlap_prefix = prefix,
        ))
        chunk_index     += 1
        current_els      = []
        current_chars    = 0
        pending_overlap  = ""   # always clear after consuming

    # ── Main loop ─────────────────────────────────────────────────────────────
    for el in element_pool:
        el_text = _element_to_text(el)
        el_len  = len(el_text)

        # ── 1. ANCHOR: Title always opens a new chunk, resets both anchors ────
        if el.element_type == "Title":
            # Natural boundary — no overlap needed
            flush(use_overlap=False)
            anchor_title   = el.content[:120]
            anchor_heading = ""
            # Start the new chunk with this Title element
            current_els.append(el)
            current_chars = el_len
            continue

        # ── 2. ANCHOR: Heading opens a new chunk, updates heading anchor ──────
        if el.element_type == "Heading":
            # Natural boundary — no overlap needed
            flush(use_overlap=False)
            anchor_heading = el.content[:120]
            # Start the new chunk with this Heading element
            current_els.append(el)
            current_chars = el_len
            continue

        # ── 3. STANDALONE: Diagram and Table always get their own chunk ───────
        if el.element_type in _STANDALONE_TYPES:
            # Flush whatever is buffered (natural boundary, no overlap)
            flush(use_overlap=False)
            # Emit the standalone element as a single-element chunk
            chunks.append(_build_chunk(
                elements       = [el],
                chunk_index    = chunk_index,
                source_file    = source_file,
                course         = course,
                anchor_title   = anchor_title,
                anchor_heading = anchor_heading,
                ocr_results    = ocr_results,
                overlap_prefix = "",   # standalone — never carry overlap
            ))
            chunk_index += 1
            # After a standalone, the next regular element starts fresh
            pending_overlap = ""
            continue

        # ── 4. OVERSIZED single element: split text, emit each segment ─────────
        # This handles a Paragraph or ListBlock that by itself exceeds HARD_MAX.
        if el_len > HARD_MAX:
            # Flush anything buffered before splitting this element
            flush(use_overlap=bool(pending_overlap))
            # Split the element text into segments
            segments = _split_long_text(el_text, HARD_MAX, OVERLAP)
            for seg_idx, (seg_text, seg_ovlp) in enumerate(segments):
                # First segment may carry pending_overlap from BEFORE this element
                # (pending_overlap was cleared by flush() above, so always "" here)
                seg_el = PageElement(
                    element_type  = el.element_type,
                    content       = seg_text,
                    page_number   = el.page_number,
                    position_hint = el.position_hint,
                    confidence    = el.confidence,
                )
                chunks.append(_build_chunk(
                    elements       = [seg_el],
                    chunk_index    = chunk_index,
                    source_file    = source_file,
                    course         = course,
                    anchor_title   = anchor_title,
                    anchor_heading = anchor_heading,
                    ocr_results    = ocr_results,
                    overlap_prefix = "",
                ))
                chunk_index += 1
                # Carry the overlap for the NEXT segment (not used after last seg)
                pending_overlap = seg_ovlp
            continue

        # ── 5. NORMAL accumulation with soft-target and hard-max logic ─────────
        projected_chars = current_chars + (2 if current_els else 0) + el_len

        if projected_chars <= SOFT_TARGET:
            # Well below soft target: keep accumulating unconditionally
            current_els.append(el)
            current_chars = projected_chars

        elif projected_chars <= HARD_MAX:
            # Between soft target and hard max.
            # We ARE above soft target but still within hard max.
            # Decide: is this a good semantic place to close the chunk?
            #
            # Close the chunk NOW (before adding el) only if el is the START
            # of a new paragraph-level idea (Paragraph, ListBlock, Equation)
            # AND the current buffer is already at or above soft target.
            # This keeps "tight" content (e.g. caption right after its equation)
            # together while still splitting at natural paragraph boundaries.
            is_natural_break = (
                el.element_type in {"Paragraph", "ListBlock", "Equation"}
                and current_chars >= SOFT_TARGET
                and current_els  # don't flush empty buffer
            )
            if is_natural_break:
                # Natural semantic boundary — flush WITHOUT overlap, start fresh
                flush(use_overlap=bool(pending_overlap))
                current_els.append(el)
                current_chars = el_len
            else:
                # Still fits and no natural break — keep accumulating
                current_els.append(el)
                current_chars = projected_chars

        else:
            # Exceeds HARD_MAX: forced split.
            #
            # Before flushing, try to find a sentence boundary inside the LAST
            # element in current_els so we can cleanly carry overlap forward.
            #
            # Compute overlap from the current buffer's text BEFORE adding el.
            current_text    = _elements_to_text(current_els)
            new_overlap     = _sentence_boundary_overlap(current_text, OVERLAP)

            # Flush the current buffer (with pending_overlap from prev hard split)
            flush(use_overlap=bool(pending_overlap))

            # Set pending_overlap so the NEXT chunk (starting with el) receives context
            pending_overlap = new_overlap

            # Start the new chunk with el
            current_els.append(el)
            current_chars = el_len

    # ── End of pool: flush any remaining elements ─────────────────────────────
    # The last chunk inherits pending_overlap if the loop ended mid-hard-split
    flush(use_overlap=bool(pending_overlap))

    # ── Post-processing: merge adjacent TINY chunks within the same anchor ────
    # A chunk is "tiny" if its total text (including header) is below min_chunk_chars.
    # Never merge across different anchor_title or anchor_heading — that would
    # corrupt the semantic boundary.
    merged: List[Chunk] = []
    for ch in chunks:
        if (
            merged
            and len(ch.text) < cfg.min_chunk_chars
            and merged[-1].anchor_title   == ch.anchor_title
            and merged[-1].anchor_heading == ch.anchor_heading
            # Never merge a standalone Diagram/Table into its neighbour
            and not (set(ch.element_types) & _STANDALONE_TYPES)
            and not (set(merged[-1].element_types) & _STANDALONE_TYPES)
        ):
            prev = merged[-1]
            merged[-1] = Chunk(
                chunk_id       = prev.chunk_id,
                text           = prev.text + "\n\n" + ch.text,
                page_start     = prev.page_start,
                page_end       = max(prev.page_end, ch.page_end),
                chunk_index    = prev.chunk_index,
                element_types  = list(dict.fromkeys(prev.element_types + ch.element_types)),
                anchor_title   = prev.anchor_title,
                anchor_heading = prev.anchor_heading,
                has_diagram    = prev.has_diagram    or ch.has_diagram,
                has_equation   = prev.has_equation   or ch.has_equation,
                has_table      = prev.has_table      or ch.has_table,
                source_file    = prev.source_file,
                course         = prev.course,
                ocr_model_used = (
                    "mixed" if prev.ocr_model_used != ch.ocr_model_used
                    else prev.ocr_model_used
                ),
                avg_confidence = round(
                    (prev.avg_confidence + ch.avg_confidence) / 2, 4
                ),
            )
        else:
            merged.append(ch)

    log.info(
        "Chunker: %d elements -> %d raw chunks -> %d after tiny-merge "
        "(soft=%d  hard=%d  overlap=%d).",
        len(element_pool), len(chunks), len(merged),
        SOFT_TARGET, HARD_MAX, OVERLAP,
    )
    return merged



# ==============================================================================
# SECTION 7 - ASYNC CONCURRENT EMBEDDER
# ==============================================================================

async def _embed_single_async(
    client,
    semaphore: asyncio.Semaphore,
    text: str,
) -> List[float]:
    """
    Embed one chunk text via asyncio.to_thread().
    Bounded by semaphore; exponential backoff on 429 rate-limit errors.
    """
    from google.genai import types
    async with semaphore:
        for attempt in range(4):
            try:
                result = await asyncio.to_thread(
                    client.models.embed_content,
                    model    = cfg.embedding_model,
                    contents = text if text.strip() else " ",
                    config   = types.EmbedContentConfig(
                        task_type             = EMBEDDING_TASK_TYPE,
                        output_dimensionality = cfg.pinecone_dimension,
                    ),
                )
                return result.embeddings[0].values
            except Exception as exc:
                if "429" in str(exc) or "quota" in str(exc).lower():
                    wait = (2 ** attempt) + (0.1 * attempt)
                    log.warning("Embed rate-limited (attempt %d) - retry in %.1fs", attempt + 1, wait)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError("Embedding failed after 4 attempts (rate limit)")


async def embed_chunks_async(chunks: List[Chunk]) -> List[List[float]]:
    """
    Embed all chunks concurrently using the group-of-batches pattern.

    Structure (example: 100 chunks, EMBED_BATCH_SIZE=10, EMBED_GROUP_SIZE=10):
        100 tasks created (one per chunk)
        -> processed in groups of 100 (EMBED_GROUP_SIZE * EMBED_BATCH_SIZE)
        -> asyncio.gather(*group) runs all tasks in the group concurrently
        -> semaphore limits real simultaneous Gemini calls to EMBED_CONCURRENCY
        -> next group starts only after current group completes

    Order is preserved: task[i] produces vector[i].
    """
    if not chunks:
        return []
    client    = _gemini_client()
    semaphore = asyncio.Semaphore(cfg.embed_concurrency)
    texts     = [c.text for c in chunks]
    n         = len(texts)

    log.info("Embedding %d chunks (concurrency=%d, batch=%d, group=%d) ...",
             n, cfg.embed_concurrency, cfg.embed_batch_size, cfg.embed_group_size)

    all_tasks = [_embed_single_async(client, semaphore, text) for text in texts]

    group_size   = cfg.embed_group_size * cfg.embed_batch_size
    all_vectors: List[List[float]] = []
    total_groups = (n + group_size - 1) // group_size

    for g_start in range(0, n, group_size):
        g_end  = min(g_start + group_size, n)
        g_num  = g_start // group_size + 1
        group  = all_tasks[g_start:g_end]
        log.info("  Embed group %d/%d: chunks %d-%d (%d tasks) ...",
                 g_num, total_groups, g_start + 1, g_end, len(group))
        vectors = await asyncio.gather(*group)
        all_vectors.extend(vectors)

    log.info("Embedded %d chunks -> %d-dim vectors.", n,
             len(all_vectors[0]) if all_vectors else 0)
    return all_vectors


def embed_chunks(chunks: List[Chunk]) -> List[List[float]]:
    """Sync entry point - runs the async embedder in a fresh event loop."""
    return asyncio.run(embed_chunks_async(chunks))


def embed_query(query_text: str) -> List[float]:
    """Embed a single search query (RETRIEVAL_QUERY task type)."""
    from google.genai import types
    client   = _gemini_client()
    response = client.models.embed_content(
        model    = cfg.embedding_model,
        contents = [query_text],
        config   = types.EmbedContentConfig(
            task_type             = EMBEDDING_QUERY_TASK_TYPE,
            output_dimensionality = cfg.pinecone_dimension,
        ),
    )
    return response.embeddings[0].values


# ==============================================================================
# SECTION 8 - VECTOR STORE
# ==============================================================================

def _pinecone_client():
    from pinecone import Pinecone
    return Pinecone(api_key=cfg.pinecone_api_key)


def get_or_create_index():
    """Return the Pinecone Index, creating it automatically if needed."""
    from pinecone import ServerlessSpec
    pc       = _pinecone_client()
    idx_name = cfg.pinecone_index_name
    existing = [i.name for i in pc.list_indexes()]
    if idx_name not in existing:
        log.info("Creating Pinecone index %s (dim=%d) ...", idx_name, cfg.pinecone_dimension)
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
    return pc.Index(idx_name)


def upsert_chunks(chunks: List[Chunk], vectors: List[List[float]]) -> int:
    """Upsert vectors to Pinecone in batches of 100. Returns upserted count."""
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) vs vectors ({len(vectors)}) mismatch.")
    index     = get_or_create_index()
    namespace = cfg.pinecone_namespace
    records   = [{"id": c.chunk_id, "values": v, "metadata": c.metadata()}
                 for c, v in zip(chunks, vectors)]
    upserted  = 0
    for start in range(0, len(records), 100):
        batch     = records[start:start + 100]
        log.info("Upserting vectors %d-%d ...", start + 1, start + len(batch))
        upserted += index.upsert(vectors=batch, namespace=namespace).upserted_count
    log.info("Upserted %d vectors total.", upserted)
    return upserted


def query_notes(
    query_vector: List[float],
    course_id:    Optional[str] = None,
    top_k:        int = 5,
    filter_dict:  Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Query Pinecone for chunks closest to query_vector."""
    index = get_or_create_index()
    pf: Dict[str, Any] = {}
    if course_id:
        pf["course_id"] = {"$eq": course_id}
    if filter_dict:
        pf.update(filter_dict)
    kwargs: Dict[str, Any] = dict(
        vector=query_vector, top_k=top_k,
        namespace=cfg.pinecone_namespace, include_metadata=True,
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


# ==============================================================================
# SECTION 9 - PIPELINE ORCHESTRATOR
# ==============================================================================

def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}m"


def run_pipeline(
    filepath:    str,
    course_id:   str,
    course_name: str,
    subject:     str = "",
    instructor:  str = "",
    semester:    str = "",
    university:  str = "",
    tags:        Optional[List[str]] = None,
) -> PipelineResult:
    """
    Run the full handwritten notes ingestion pipeline (v2).

    Stage 1 - Ingest:   load PDF/image pages as PIL images
    Stage 2 - OCR:      async concurrent Flash/Pro per page -> sorted PageOCRResult list
    Stage 2b - Pool:    collect all PageElements in page-number order
    Stage 3 - Chunk:    cross-page heading-anchored semantic chunking
    Stage 4 - Embed:    async concurrent group embedding with semaphore + backoff
    Stage 5 - Upsert:   batched Pinecone upsert at 100 vectors/call
    """
    cfg.validate()
    course = CourseMetadata(
        course_id=course_id, course_name=course_name,
        subject=subject, instructor=instructor,
        semester=semester, university=university, tags=tags or [],
    )
    result  = PipelineResult(source_file=Path(filepath).name, course=course)
    t_total = time.time()

    log.info("=" * 62)
    log.info("Pipeline v2: %s  course=%s", result.source_file, course_id)
    log.info("=" * 62)

    # Stage 1 - Ingest
    t = time.time()
    log.info("[1/5] Ingest ...")
    try:
        pages = ingest(filepath)
        log.info("      %d page(s) in %s.", len(pages), _elapsed(t))
    except Exception as exc:
        log.error("[1/5] Ingest failed: %s", exc)
        result.errors.append(f"Ingest failed: {exc}")
        return result

    # Stage 2 - Async concurrent OCR (all pages simultaneously)
    t = time.time()
    log.info("[2/5] OCR - %d pages concurrently (OCR_CONCURRENCY=%d) ...",
             len(pages), cfg.ocr_concurrency)
    try:
        ocr_results, flash_c, pro_c, ocr_errors = asyncio.run(
            ocr_all_pages_async(pages)
        )
    except Exception as exc:
        log.error("[2/5] OCR stage failed: %s", exc)
        result.errors.append(f"OCR stage failed: {exc}")
        return result

    result.ocr_results = ocr_results
    result.flash_pages = flash_c
    result.pro_pages   = pro_c
    result.errors.extend(ocr_errors)
    log.info("      OCR done in %s  Flash=%d  Pro=%d  Errors=%d",
             _elapsed(t), flash_c, pro_c, len(ocr_errors))

    if not ocr_results:
        result.errors.append("All pages failed OCR.")
        return result

    # Stage 2b - Build element pool (all pages, sorted by page_number)
    element_pool: List[PageElement] = []
    for ocr in ocr_results:   # already sorted by page_number in ocr_all_pages_async
        element_pool.extend(ocr.elements)
    result.element_pool = element_pool
    log.info("      Element pool: %d elements from %d pages.",
             len(element_pool), len(ocr_results))

    # Stage 3 - Cross-page chunking
    t = time.time()
    log.info("[3/5] Cross-page chunking (%d elements) ...", len(element_pool))
    result.chunks = chunk_element_pool(element_pool, course, filepath, ocr_results)
    log.info("      %d chunks in %s.", len(result.chunks), _elapsed(t))
    if not result.chunks:
        log.warning("No chunks produced - pages may have been blank.")
        return result

    # Stage 4 - Async concurrent embedding
    t = time.time()
    log.info("[4/5] Embedding %d chunks (EMBED_CONCURRENCY=%d) ...",
             len(result.chunks), cfg.embed_concurrency)
    try:
        vectors = embed_chunks(result.chunks)
        log.info("      Embedding done in %s.", _elapsed(t))
    except Exception as exc:
        log.error("[4/5] Embedding failed: %s", exc)
        result.errors.append(f"Embedding failed: {exc}")
        return result

    # Stage 5 - Pinecone upsert
    t = time.time()
    log.info("[5/5] Upserting to Pinecone ...")
    try:
        result.upserted_count = upsert_chunks(result.chunks, vectors)
        log.info("      Upserted %d vectors in %s.", result.upserted_count, _elapsed(t))
    except Exception as exc:
        log.error("[5/5] Upsert failed: %s", exc)
        result.errors.append(f"Upsert failed: {exc}")

    log.info("=" * 62)
    log.info("Pipeline v2 done in %s.", _elapsed(t_total))
    log.info("=" * 62)
    print(result.summary())
    return result


# ==============================================================================
# SECTION 10 - CLI
# ==============================================================================

def _cli_ingest(args: argparse.Namespace) -> None:
    run_pipeline(
        filepath    = args.filepath,
        course_id   = args.course_id or "",
        course_name = args.course_name or "",
        subject     = args.subject or "",
        instructor  = args.instructor or "",
        semester    = args.semester or "",
        university  = args.university or "",
        tags        = args.tags or [],
    )


def _cli_query(args: argparse.Namespace) -> None:
    print(f"\nSearching: {args.query!r} ...\n")
    vec     = embed_query(args.query)
    results = query_notes(vec, course_id=args.course_id or None, top_k=args.top_k)
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"[{i}] Score: {r['score']:.4f}")
        print(f"     Course  : {m.get('course_id')} - {m.get('course_name')}")
        print(f"     Pages   : {m.get('page_start')}-{m.get('page_end')}")
        print(f"     Anchor  : {m.get('anchor_title', '')} > {m.get('anchor_heading', '')}")
        print(f"     Elements: {m.get('element_types', '')}")
        print(f"     Model   : {m.get('ocr_model_used')}  Conf: {m.get('avg_confidence')}")
        print(f"     Preview : {m.get('text_preview', '')[:200]}")
        print()


def _cli_stats(_args: argparse.Namespace) -> None:
    print(json.dumps(index_stats(), indent=2))


def _ingest_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--course-id",   required=False)
    p.add_argument("--course-name", default="")
    p.add_argument("--subject",     default="")
    p.add_argument("--instructor",  default="")
    p.add_argument("--semester",    default="")
    p.add_argument("--university",  default="")
    p.add_argument("--tags", nargs="*", default=[])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingestion_pipeline.py",
        description="Handwritten Notes Ingestion Pipeline v2 - Gemini + Pinecone",
    )
    sub = parser.add_subparsers(dest="command")
    ing = sub.add_parser("ingest", help="Ingest a PDF or image file.")
    ing.add_argument("filepath")
    _ingest_flags(ing)
    q = sub.add_parser("query", help="Semantic search over ingested notes.")
    q.add_argument("query")
    q.add_argument("--course-id")
    q.add_argument("--top-k", type=int, default=5)
    sub.add_parser("stats", help="Show Pinecone index statistics.")
    parser.add_argument("filepath", nargs="?")
    _ingest_flags(parser)
    return parser


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
