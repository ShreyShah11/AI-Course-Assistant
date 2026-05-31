"""
Academic Course Content Chunking Pipeline
==========================================
Designed to plug directly onto TesseractOCREngine output (OCRResult).

Handles:
  - Typed academic text (paragraphs, headings, definitions, theorems)
  - Source code blocks (Python, Java, C++, SQL, pseudocode, …)
  - Mixed content pages (lecture slides, textbook chapters)
  - Tables, formulas, captions, numbered lists
  - Multi-page documents with cross-page context

Chunking strategy:
  - Prose      → Semantic / paragraph-boundary chunking with overlap
  - Code        → Whole-block chunking (never split a code block)
  - Headings    → Anchor metadata, not standalone chunks
  - Formulas    → Atomic chunks (never split)
  - Lists       → Group-then-split: list items grouped under their heading
  - Tables      → Whole-table chunks with structured metadata

Every chunk carries rich metadata for retrieval scoring, reranking,
and LLM context injection.
"""

import re
import hashlib
import logging
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ChunkType(Enum):
    PROSE        = "prose"        # General academic text / paragraphs
    CODE         = "code"         # Source code block (whole)
    FORMULA      = "formula"      # Mathematical expression
    LIST         = "list"         # Bullet / numbered list group
    TABLE        = "table"        # Tabular content
    HEADING      = "heading"      # Section heading (usually merged into prose)
    DEFINITION   = "definition"   # "Definition:", "Theorem:", etc.
    EXAMPLE      = "example"      # "Example:", "e.g." blocks
    CAPTION      = "caption"      # Figure / table caption
    MIXED        = "mixed"        # Heading + following text (slide-style)


class ContentSignal(Enum):
    """Semantic markers detected inside text, used for retrieval boosting."""
    DEFINITION   = "definition"
    THEOREM      = "theorem"
    PROOF        = "proof"
    EXAMPLE      = "example"
    ALGORITHM    = "algorithm"
    EXERCISE     = "exercise"
    NOTE         = "note"
    WARNING      = "warning"
    SUMMARY      = "summary"
    REFERENCE    = "reference"


# ─────────────────────────────────────────────
# Metadata dataclass
# ─────────────────────────────────────────────

@dataclass
class ChunkMetadata:
    # ── Identity
    chunk_id:           str   = ""          # SHA-256 of content (dedup key)
    chunk_index:        int   = 0           # Position in document
    chunk_type:         str   = ""          # ChunkType.value

    # ── Source document
    source_file:        str   = ""          # Original filename / path
    document_id:        str   = ""          # SHA-256 of full document text
    page_numbers:       list  = field(default_factory=list)  # Pages this chunk spans
    total_pages:        int   = 0

    # ── Course / academic context
    course_id:          str   = ""
    course_name:        str   = ""
    subject_area:       str   = ""          # e.g. "algorithms", "databases"
    document_title:     str   = ""          # Inferred from first heading
    chapter:            str   = ""          # Chapter heading if present
    section:            str   = ""          # Section heading if present
    subsection:         str   = ""          # Sub-section heading if present

    # ── Positional
    heading_path:       list  = field(default_factory=list)   # Breadcrumb
    prev_chunk_id:      str   = ""
    next_chunk_id:      str   = ""
    is_continuation:    bool  = False       # True if split from larger block
    part_index:         int   = 0          # Which part of a split block

    # ── Content
    token_count:        int   = 0          # Approx token count (words × 1.3)
    char_count:         int   = 0
    word_count:         int   = 0
    line_count:         int   = 0
    language:           str   = ""         # Code language if chunk_type=code
    content_signals:    list  = field(default_factory=list)   # ContentSignal values
    keywords:           list  = field(default_factory=list)   # Top TF nouns

    # ── OCR quality
    ocr_confidence:     float = 0.0        # Mean Tesseract confidence (0-100)
    image_type:         str   = ""         # e.g. "photo", "clean_scan"
    skew_corrected:     bool  = False
    skew_angle:         float = 0.0
    ocr_warnings:       list  = field(default_factory=list)
    preprocessing_stages: list = field(default_factory=list)

    # ── Retrieval hints
    has_code:           bool  = False
    has_formula:        bool  = False
    has_list:           bool  = False
    is_definition:      bool  = False
    is_example:         bool  = False
    semantic_density:   float = 0.0        # Unique words / total words
    context_window:     str   = ""         # Heading + first 200 chars for hybrid search

    # ── Timestamps
    created_at:         str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
# Chunk dataclass
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    text:       str
    metadata:   ChunkMetadata

    def to_dict(self) -> dict:
        return {
            "text":     self.text,
            "metadata": self.metadata.to_dict()
        }

    def __repr__(self):
        meta = self.metadata
        return (
            f"Chunk(type={meta.chunk_type}, page={meta.page_numbers}, "
            f"tokens≈{meta.token_count}, section={meta.section!r})"
        )


# ─────────────────────────────────────────────
# Semantic signal detector
# ─────────────────────────────────────────────

class SemanticSignalDetector:
    """
    Detects academic semantic markers in text.
    Used to enrich metadata and boost retrieval relevance.
    """

    PATTERNS: dict[ContentSignal, list[str]] = {
        ContentSignal.DEFINITION: [
            r'^\s*(?:Definition|Defn|Def)\s*[\d.]*\s*[:\.]',
            r'\b(?:is defined as|is called|we define|denote by)\b',
            r'\biff\b',                     # "if and only if"
        ],
        ContentSignal.THEOREM: [
            r'^\s*(?:Theorem|Lemma|Corollary|Proposition|Claim)\s*[\d.]*\s*[:\.]',
        ],
        ContentSignal.PROOF: [
            r'^\s*(?:Proof|Pf)\s*[:\.]',
            r'\b(?:Q\.E\.D|□|∎|QED)\b',
            r'\bby induction\b', r'\bbase case\b', r'\binductive step\b',
        ],
        ContentSignal.EXAMPLE: [
            r'^\s*(?:Example|Ex|e\.g\.)',
            r'\bfor example\b', r'\bfor instance\b',
            r'\bconsider the following\b',
        ],
        ContentSignal.ALGORITHM: [
            r'^\s*(?:Algorithm|Algo)\s*[\d.]*\s*[:\.]',
            r'\b(?:Input|Output|Procedure|Subroutine)\s*:',
            r'\btime complexity\b', r'\bspace complexity\b',
            r'\bO\s*\(', r'\bΘ\s*\(', r'\bΩ\s*\(',
        ],
        ContentSignal.EXERCISE: [
            r'^\s*(?:Exercise|Problem|Q\.|Question)\s*[\d.]+',
            r'\bsolve the following\b', r'\bprove that\b',
        ],
        ContentSignal.NOTE: [
            r'^\s*(?:Note|Remark|Observation|NB)\s*[:\.]',
        ],
        ContentSignal.WARNING: [
            r'^\s*(?:Warning|Caution|Important)\s*[:\.]',
            r'\bcommon mistake\b', r'\bbeware\b',
        ],
        ContentSignal.SUMMARY: [
            r'^\s*(?:Summary|Conclusion|Key (?:Points|Takeaways))\s*[:\.]',
            r'\bin summary\b', r'\bto summarize\b',
        ],
        ContentSignal.REFERENCE: [
            r'\[[\d,\s]+\]',               # Citation like [1], [1,2]
            r'\b(?:see|refer to|cf\.)\s+(?:section|chapter|figure|table)\b',
        ],
    }

    @classmethod
    def detect(cls, text: str) -> list[str]:
        found = []
        for signal, patterns in cls.PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
                    found.append(signal.value)
                    break
        return found


# ─────────────────────────────────────────────
# Keyword extractor (lightweight, no ML)
# ─────────────────────────────────────────────

class KeywordExtractor:
    """
    Extracts top N keywords from a chunk using TF heuristics.
    No external NLP models — runs entirely on regex + frequency.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "shall", "should", "may", "might", "can", "could",
        "this", "that", "these", "those", "it", "its", "we", "they", "them",
        "their", "our", "us", "you", "your", "he", "she", "his", "her",
        "not", "no", "if", "then", "else", "when", "where", "which", "who",
        "how", "what", "as", "so", "such", "each", "any", "all", "both",
        "more", "most", "some", "into", "through", "over", "also", "than",
        "other", "same", "after", "before", "following", "given", "let",
        "using", "use", "used", "show", "shown", "consider", "note",
    }

    @classmethod
    def extract(cls, text: str, top_n: int = 8) -> list[str]:
        # Tokenize: only proper words ≥ 3 chars, ignore numbers/symbols
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z_\-]{2,}\b', text.lower())
        tokens = [t for t in tokens if t not in cls.STOPWORDS]

        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        # Bonus for capitalized terms (likely proper nouns / technical terms)
        cap_tokens = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
        for t in cap_tokens:
            freq[t.lower()] = freq.get(t.lower(), 0) + 2

        sorted_kw = sorted(freq, key=freq.get, reverse=True)
        return sorted_kw[:top_n]


# ─────────────────────────────────────────────
# Heading tracker
# ─────────────────────────────────────────────

class HeadingTracker:
    """
    Maintains a running stack of document headings as chunks are processed.
    Provides breadcrumb-style section paths for each chunk.
    """

    # Heuristic levels: level 1 = chapter, 2 = section, 3 = subsection
    LEVEL_PATTERNS = [
        (1, re.compile(r'^\s*(?:chapter|part|unit)\s*\d+', re.IGNORECASE)),
        (1, re.compile(r'^\s*\d+\.\s+[A-Z]')),         # "1. Introduction"
        (2, re.compile(r'^\s*\d+\.\d+\s+[A-Z]')),      # "1.2 Background"
        (3, re.compile(r'^\s*\d+\.\d+\.\d+\s+[A-Z]')), # "1.2.3 Details"
    ]

    def __init__(self):
        self.stack: list[tuple[int, str]] = []  # (level, heading_text)

    def update(self, heading_text: str) -> None:
        level = self._infer_level(heading_text)
        # Pop deeper or equal levels before pushing
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, heading_text.strip()))

    def _infer_level(self, text: str) -> int:
        for level, pat in self.LEVEL_PATTERNS:
            if pat.match(text):
                return level
        # Fallback: all-caps or title-case short text → section
        if text.isupper() or (len(text) < 60 and text.istitle()):
            return 2
        return 3

    @property
    def breadcrumb(self) -> list[str]:
        return [h for _, h in self.stack]

    @property
    def chapter(self) -> str:
        for level, h in self.stack:
            if level == 1:
                return h
        return ""

    @property
    def section(self) -> str:
        for level, h in self.stack:
            if level == 2:
                return h
        return ""

    @property
    def subsection(self) -> str:
        for level, h in reversed(self.stack):
            if level == 3:
                return h
        return ""


# ─────────────────────────────────────────────
# Text splitter
# ─────────────────────────────────────────────

class SemanticTextSplitter:
    """
    Splits prose text at natural semantic boundaries:
    paragraph > sentence > word.

    Uses overlap to preserve context across chunk boundaries.
    """

    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 60):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """word count × 1.3 ≈ BPE token count for English academic text."""
        return int(len(text.split()) * 1.3)

    def split(self, text: str) -> list[str]:
        """
        Split text into chunks ≤ max_tokens with overlap_tokens overlap.
        Tries paragraph → sentence → word boundaries in that priority order.
        """
        if self._approx_tokens(text) <= self.max_tokens:
            return [text]

        # Try paragraph splits first (double newline)
        paragraphs = re.split(r'\n{2,}', text)
        if len(paragraphs) > 1:
            return self._merge_splits(paragraphs)

        # Fall back to sentence splits
        sentences = self._split_sentences(text)
        if len(sentences) > 1:
            return self._merge_splits(sentences)

        # Last resort: word-level split
        return self._word_split(text)

    def _split_sentences(self, text: str) -> list[str]:
        """
        Sentence splitter aware of:
        - Abbreviations (e.g., Dr., Fig., vs.)
        - Code-like content (skip splitting inside backtick / code fence regions)
        - Numbered lists
        """
        # Protect known abbreviations from splitting
        abbreviations = r'(?:Dr|Mr|Mrs|Ms|Prof|Fig|Eq|vs|et al|i\.e|e\.g|approx|est|ref)'
        protected = re.sub(rf'\b({abbreviations})\.\s', r'\1@@@ ', text)
        # Split on . ! ? followed by space + uppercase
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
        # Restore abbreviation dots
        return [p.replace('@@@', '.') for p in parts if p.strip()]

    def _merge_splits(self, parts: list[str]) -> list[str]:
        """
        Greedily merge parts into chunks ≤ max_tokens,
        then prepend the last overlap_tokens of previous chunk to next chunk.
        """
        chunks = []
        current_parts = []
        current_tokens = 0
        overlap_text = ""

        for part in parts:
            part_tokens = self._approx_tokens(part)

            # Single part exceeds limit — must split at word level
            if part_tokens > self.max_tokens:
                if current_parts:
                    chunks.append(self._join(current_parts))
                    overlap_text = self._tail_words(self._join(current_parts),
                                                     self.overlap_tokens)
                for sub in self._word_split(part):
                    chunks.append(overlap_text + sub if overlap_text else sub)
                    overlap_text = self._tail_words(sub, self.overlap_tokens)
                current_parts = []
                current_tokens = 0
                continue

            if current_tokens + part_tokens > self.max_tokens:
                if current_parts:
                    chunk_text = self._join(current_parts)
                    chunks.append(
                        (overlap_text + "\n\n" + chunk_text).strip()
                        if overlap_text else chunk_text
                    )
                    overlap_text = self._tail_words(chunk_text, self.overlap_tokens)
                current_parts = [part]
                current_tokens = part_tokens
            else:
                current_parts.append(part)
                current_tokens += part_tokens

        if current_parts:
            chunk_text = self._join(current_parts)
            chunks.append(
                (overlap_text + "\n\n" + chunk_text).strip()
                if overlap_text else chunk_text
            )

        return [c for c in chunks if c.strip()]

    def _word_split(self, text: str) -> list[str]:
        words = text.split()
        overlap_words = max(1, int(self.overlap_tokens / 1.3))
        chunks = []
        chunk_words_max = max(1, int(self.max_tokens / 1.3))
        start = 0
        while start < len(words):
            end = min(start + chunk_words_max, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - overlap_words
        return chunks

    @staticmethod
    def _join(parts: list[str]) -> str:
        return "\n\n".join(p.strip() for p in parts if p.strip())

    @staticmethod
    def _tail_words(text: str, target_tokens: int) -> str:
        words = text.split()
        n = max(1, int(target_tokens / 1.3))
        return " ".join(words[-n:]) if len(words) > n else text


# ─────────────────────────────────────────────
# ID generator
# ─────────────────────────────────────────────

def _chunk_id(text: str, index: int) -> str:
    payload = f"{index}:{text[:200]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _doc_id(full_text: str) -> str:
    return hashlib.sha256(full_text.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────
# Main chunking pipeline
# ─────────────────────────────────────────────

class ChunkingPipeline:
    """
    Converts OCRResult (from TesseractOCREngine) into a list of Chunk
    objects, each with rich metadata suitable for vector store ingestion
    and hybrid BM25 + semantic retrieval.

    Chunking strategy per block type
    ─────────────────────────────────
    HEADING     → Update HeadingTracker; merge into following block if short
    PARAGRAPH   → SemanticTextSplitter with overlap
    CODE        → One chunk per code block, never split
    FORMULA     → Atomic chunk, never split
    LIST_ITEM   → Group consecutive list items under same heading, then split
    TABLE_CELL  → One chunk per logical table group
    CAPTION     → Merge into preceding figure/table chunk; fallback standalone
    DEFINITION  → Atomic chunk (signal detected, never split mid-definition)

    Usage
    ─────
        from ocr_engine import TesseractOCREngine
        from chunker import ChunkingPipeline

        engine = TesseractOCREngine()
        ocr_result = engine.run("lecture_slide.jpg")

        pipeline = ChunkingPipeline(
            source_file="lec03_sorting.pdf",
            course_id="CS201",
            course_name="Data Structures & Algorithms",
            subject_area="sorting",
        )
        chunks = pipeline.run([ocr_result])   # list of OCRResult (multi-page)

        # Ready for vector store:
        for chunk in chunks:
            print(chunk.text)
            print(chunk.metadata.to_dict())
    """

    def __init__(
        self,
        source_file:    str   = "",
        course_id:      str   = "",
        course_name:    str   = "",
        subject_area:   str   = "",
        max_tokens:     int   = 400,    # Max tokens per prose chunk
        overlap_tokens: int   = 60,     # Overlap between consecutive prose chunks
        min_chunk_tokens: int = 20,     # Discard chunks smaller than this
        merge_headings: bool  = True,   # Prepend headings to following chunk text
        debug:          bool  = False,
    ):
        self.source_file      = source_file
        self.course_id        = course_id
        self.course_name      = course_name
        self.subject_area     = subject_area
        self.max_tokens       = max_tokens
        self.overlap_tokens   = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.merge_headings   = merge_headings
        self.debug            = debug

        self.splitter         = SemanticTextSplitter(max_tokens, overlap_tokens)
        self.heading_tracker  = HeadingTracker()

    # ─────────────────────────────────────────
    # Primary entry point
    # ─────────────────────────────────────────

    def run(self, ocr_results: list) -> list[Chunk]:
        """
        Accept a list of OCRResult objects (one per page).
        Returns a flat, ordered list of Chunk objects.

        Parameters
        ----------
        ocr_results : list[OCRResult]
            Output of TesseractOCREngine.run() for each page.
        """
        if not ocr_results:
            return []

        # Build full-document text for doc-level ID and title inference
        full_doc_text = "\n\n".join(r.full_text for r in ocr_results)
        doc_id        = _doc_id(full_doc_text)
        total_pages   = len(ocr_results)

        # Infer document title from first heading-like block on page 1
        document_title = self._infer_document_title(ocr_results[0])

        # Process page by page
        raw_chunks: list[Chunk] = []
        pending_heading: Optional[str] = None  # Heading waiting to be merged
        pending_list_items: list[str]  = []    # Accumulating list items
        pending_list_page: int         = 1
        pending_list_conf: float       = 100.0

        for ocr_result in ocr_results:
            page = ocr_result.page

            # Base metadata shared by all chunks on this page
            page_base = self._base_metadata(
                ocr_result, doc_id, total_pages, document_title
            )

            blocks = ocr_result.blocks  # Already sorted top-to-bottom

            for block_idx, block in enumerate(blocks):
                from_ocr = block  # TextBlock from ocr_engine.py
                text      = block.text.strip()
                if not text:
                    continue

                btype_str = block.block_type.value   # BlockType enum → str

                # ── HEADING ──────────────────────────────────────────────
                if btype_str == "heading":
                    # Flush any pending list before a new heading
                    if pending_list_items:
                        raw_chunks.extend(
                            self._make_list_chunk(
                                pending_list_items,
                                pending_list_page, pending_list_conf,
                                page_base, doc_id, total_pages, document_title,
                                ocr_result, pending_heading
                            )
                        )
                        pending_list_items = []

                    self.heading_tracker.update(text)
                    pending_heading = text

                    if self.debug:
                        logger.debug(f"Heading: {text!r}")

                    # Short headings are merged into the next block, not standalone
                    continue

                # ── LIST ITEM ─────────────────────────────────────────────
                if btype_str == "list_item":
                    pending_list_items.append(text)
                    pending_list_page = page
                    pending_list_conf = min(pending_list_conf, block.confidence)
                    continue

                # Flush pending list if next block is NOT a list item
                if pending_list_items:
                    raw_chunks.extend(
                        self._make_list_chunk(
                            pending_list_items,
                            pending_list_page, pending_list_conf,
                            page_base, doc_id, total_pages, document_title,
                            ocr_result, pending_heading
                        )
                    )
                    pending_list_items = []

                # ── CODE ──────────────────────────────────────────────────
                if btype_str == "code":
                    chunk = self._make_code_chunk(
                        block, pending_heading,
                        page_base, doc_id, total_pages, document_title, ocr_result
                    )
                    raw_chunks.append(chunk)
                    pending_heading = None  # Consumed
                    continue

                # ── FORMULA ───────────────────────────────────────────────
                if btype_str == "formula":
                    chunk = self._make_formula_chunk(
                        block, pending_heading,
                        page_base, doc_id, total_pages, document_title, ocr_result
                    )
                    raw_chunks.append(chunk)
                    pending_heading = None
                    continue

                # ── CAPTION ───────────────────────────────────────────────
                if btype_str == "caption":
                    chunk = self._make_caption_chunk(
                        block, pending_heading,
                        page_base, doc_id, total_pages, document_title, ocr_result
                    )
                    raw_chunks.append(chunk)
                    continue

                # ── PARAGRAPH / TABLE_CELL / UNKNOWN (prose path) ─────────
                #
                # Check for definition / example signal BEFORE splitting
                # (never split mid-definition)
                signals = SemanticSignalDetector.detect(text)
                is_definition = "definition" in signals or "theorem" in signals
                is_example    = "example" in signals

                # Prepend heading as context if merging is enabled
                contextual_text = text
                if self.merge_headings and pending_heading:
                    # Only prepend if heading not already present in text
                    if pending_heading.strip() not in text[:100]:
                        contextual_text = f"{pending_heading}\n\n{text}"
                    pending_heading = None

                # Atomic chunks (never split)
                if is_definition:
                    chunk = self._make_prose_chunk(
                        contextual_text, block, ChunkType.DEFINITION,
                        signals, page_base, doc_id, total_pages,
                        document_title, ocr_result
                    )
                    raw_chunks.append(chunk)
                    continue

                # Split large prose blocks
                parts = self.splitter.split(contextual_text)
                for part_idx, part in enumerate(parts):
                    ctype = ChunkType.EXAMPLE if is_example else ChunkType.PROSE
                    chunk = self._make_prose_chunk(
                        part, block, ctype, signals,
                        page_base, doc_id, total_pages,
                        document_title, ocr_result,
                        part_index=part_idx,
                        is_continuation=(part_idx > 0)
                    )
                    raw_chunks.append(chunk)

                pending_heading = None

        # Flush any remaining list items at end of document
        if pending_list_items and ocr_results:
            last_result = ocr_results[-1]
            page_base = self._base_metadata(
                last_result, doc_id, total_pages, document_title
            )
            raw_chunks.extend(
                self._make_list_chunk(
                    pending_list_items,
                    last_result.page, 80.0,
                    page_base, doc_id, total_pages, document_title,
                    last_result, pending_heading
                )
            )

        # ── Post-process: filter, link, index ────────────────────────────
        chunks = self._post_process(raw_chunks)

        logger.info(
            f"Chunking complete: {len(chunks)} chunks from "
            f"{total_pages} page(s) of '{self.source_file}'"
        )
        return chunks

    # ─────────────────────────────────────────
    # Chunk factories
    # ─────────────────────────────────────────

    def _make_prose_chunk(
        self,
        text:           str,
        block,                           # TextBlock
        chunk_type:     ChunkType,
        signals:        list[str],
        base_meta:      dict,
        doc_id:         str,
        total_pages:    int,
        document_title: str,
        ocr_result,                      # OCRResult
        part_index:     int = 0,
        is_continuation: bool = False,
    ) -> Chunk:
        meta = self._build_metadata(
            text=text,
            chunk_type=chunk_type,
            block=block,
            ocr_result=ocr_result,
            base_meta=base_meta,
            doc_id=doc_id,
            total_pages=total_pages,
            document_title=document_title,
            signals=signals,
            part_index=part_index,
            is_continuation=is_continuation,
        )
        return Chunk(text=text, metadata=meta)

    def _make_code_chunk(
        self, block, pending_heading: Optional[str],
        base_meta, doc_id, total_pages, document_title, ocr_result
    ) -> Chunk:
        lang = block.language.value if block.language else "unknown"

        # Prepend a context comment so the LLM knows what this code is about
        header = ""
        if pending_heading:
            header = f"# Context: {pending_heading}\n"

        text = header + block.text if header else block.text
        signals = SemanticSignalDetector.detect(block.text)
        # Algorithms are common in code blocks
        if re.search(r'\b(algorithm|sort|search|tree|graph|dp|hash)\b',
                     block.text, re.IGNORECASE):
            signals.append(ContentSignal.ALGORITHM.value)

        meta = self._build_metadata(
            text=text,
            chunk_type=ChunkType.CODE,
            block=block,
            ocr_result=ocr_result,
            base_meta=base_meta,
            doc_id=doc_id,
            total_pages=total_pages,
            document_title=document_title,
            signals=list(set(signals)),
            extra={"language": lang, "has_code": True},
        )
        return Chunk(text=text, metadata=meta)

    def _make_formula_chunk(
        self, block, pending_heading: Optional[str],
        base_meta, doc_id, total_pages, document_title, ocr_result
    ) -> Chunk:
        # Prepend heading so formula is interpretable without surrounding context
        text = block.text
        if pending_heading:
            text = f"[Formula in: {pending_heading}]\n{text}"

        meta = self._build_metadata(
            text=text,
            chunk_type=ChunkType.FORMULA,
            block=block,
            ocr_result=ocr_result,
            base_meta=base_meta,
            doc_id=doc_id,
            total_pages=total_pages,
            document_title=document_title,
            signals=[],
            extra={"has_formula": True},
        )
        return Chunk(text=text, metadata=meta)

    def _make_caption_chunk(
        self, block, pending_heading: Optional[str],
        base_meta, doc_id, total_pages, document_title, ocr_result
    ) -> Chunk:
        meta = self._build_metadata(
            text=block.text,
            chunk_type=ChunkType.CAPTION,
            block=block,
            ocr_result=ocr_result,
            base_meta=base_meta,
            doc_id=doc_id,
            total_pages=total_pages,
            document_title=document_title,
            signals=[],
        )
        return Chunk(text=block.text, metadata=meta)

    def _make_list_chunk(
        self,
        items:          list[str],
        page:           int,
        conf:           float,
        base_meta:      dict,
        doc_id:         str,
        total_pages:    int,
        document_title: str,
        ocr_result,
        pending_heading: Optional[str],
    ) -> list["Chunk"]:
        if not items:
            return []

        # Prefix list with heading for context
        header = f"{pending_heading}\n\n" if pending_heading else ""
        list_text = header + "\n".join(items)

        signals = SemanticSignalDetector.detect(list_text)

        # Split if very long list
        parts = self.splitter.split(list_text)
        chunks = []
        for i, part in enumerate(parts):
            # Create a minimal fake block-like object for metadata building
            class _FakeBlock:
                confidence  = conf
                bbox        = (0, 0, 0, 0)
                line_count  = len(items)
                is_code     = False
                language    = None

            meta = self._build_metadata(
                text=part,
                chunk_type=ChunkType.LIST,
                block=_FakeBlock(),
                ocr_result=ocr_result,
                base_meta=base_meta,
                doc_id=doc_id,
                total_pages=total_pages,
                document_title=document_title,
                signals=signals,
                part_index=i,
                is_continuation=(i > 0),
                extra={"has_list": True},
            )
            chunks.append(Chunk(text=part, metadata=meta))
        return chunks

    # ─────────────────────────────────────────
    # Metadata builder
    # ─────────────────────────────────────────

    def _build_metadata(
        self,
        text:           str,
        chunk_type:     ChunkType,
        block,
        ocr_result,
        base_meta:      dict,
        doc_id:         str,
        total_pages:    int,
        document_title: str,
        signals:        list[str],
        part_index:     int  = 0,
        is_continuation: bool = False,
        extra:          dict = None,
    ) -> ChunkMetadata:
        extra = extra or {}
        words     = text.split()
        word_count = len(words)
        token_count = int(word_count * 1.3)
        char_count  = len(text)
        line_count  = text.count('\n') + 1

        unique_words = set(w.lower() for w in words if w.isalpha())
        semantic_density = len(unique_words) / max(1, word_count)

        keywords = KeywordExtractor.extract(text)

        # Context window: heading path + first 200 chars of text
        breadcrumb  = " > ".join(self.heading_tracker.breadcrumb) if self.heading_tracker.breadcrumb else ""
        context_win = (breadcrumb + "\n" + text[:200]).strip() if breadcrumb else text[:200]

        meta = ChunkMetadata(
            # Identity
            chunk_id         = "",          # Set in post_process after indexing
            chunk_index      = 0,           # Set in post_process
            chunk_type       = chunk_type.value,

            # Source
            source_file      = self.source_file,
            document_id      = doc_id,
            page_numbers     = [ocr_result.page],
            total_pages      = total_pages,

            # Course
            course_id        = self.course_id,
            course_name      = self.course_name,
            subject_area     = self.subject_area,
            document_title   = document_title,
            chapter          = self.heading_tracker.chapter,
            section          = self.heading_tracker.section,
            subsection       = self.heading_tracker.subsection,

            # Positional
            heading_path     = self.heading_tracker.breadcrumb.copy(),
            is_continuation  = is_continuation,
            part_index       = part_index,

            # Content
            token_count      = token_count,
            char_count       = char_count,
            word_count       = word_count,
            line_count       = line_count,
            language         = extra.get("language",
                               block.language.value
                               if getattr(block, "language", None) else ""),
            content_signals  = signals,
            keywords         = keywords,

            # OCR quality
            ocr_confidence   = round(block.confidence, 2),
            image_type       = ocr_result.image_type,
            skew_corrected   = ocr_result.skew_corrected,
            skew_angle       = round(ocr_result.skew_angle, 2),
            ocr_warnings     = list(ocr_result.warnings),
            preprocessing_stages = list(ocr_result.stages_applied),

            # Retrieval hints
            has_code         = extra.get("has_code", getattr(block, "is_code", False)),
            has_formula      = extra.get("has_formula", chunk_type == ChunkType.FORMULA),
            has_list         = extra.get("has_list", chunk_type == ChunkType.LIST),
            is_definition    = "definition" in signals or "theorem" in signals,
            is_example       = "example" in signals,
            semantic_density = round(semantic_density, 3),
            context_window   = context_win,

            created_at       = datetime.now(timezone.utc).isoformat(),
        )
        return meta

    # ─────────────────────────────────────────
    # Helper: base metadata (page-level)
    # ─────────────────────────────────────────

    def _base_metadata(self, ocr_result, doc_id, total_pages, document_title) -> dict:
        """Returns a dict used as a template (not a ChunkMetadata object)."""
        return {
            "doc_id":         doc_id,
            "total_pages":    total_pages,
            "document_title": document_title,
        }

    # ─────────────────────────────────────────
    # Helper: document title inference
    # ─────────────────────────────────────────

    def _infer_document_title(self, first_ocr_result) -> str:
        """
        Look at the first page's blocks for the most title-like string:
        - Block at top of page
        - All-caps or title-case
        - Shorter than 120 chars
        """
        if not first_ocr_result.blocks:
            return ""

        for block in first_ocr_result.blocks[:5]:  # Check first 5 blocks
            text = block.text.strip()
            if not text:
                continue
            if len(text) < 120 and block.bbox[1] < 300:  # Near top of page
                if text.isupper() or text.istitle() or block.block_type.value == "heading":
                    return text

        # Fallback: first non-empty block text (truncated)
        for block in first_ocr_result.blocks:
            if block.text.strip():
                return block.text.strip()[:80]
        return ""

    # ─────────────────────────────────────────
    # Post-processing: filter + link + ID
    # ─────────────────────────────────────────

    def _post_process(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        1. Filter out too-short chunks
        2. Assign sequential chunk_index
        3. Compute chunk_id (hash)
        4. Link prev/next chunk IDs
        5. Deduplicate (exact content hash)
        """
        # Filter short chunks
        filtered = [
            c for c in chunks
            if SemanticTextSplitter._approx_tokens(c.text) >= self.min_chunk_tokens
        ]

        # Deduplicate by text hash
        seen_hashes: set[str] = set()
        deduped = []
        for chunk in filtered:
            h = hashlib.sha256(chunk.text.strip().encode()).hexdigest()[:16]
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(chunk)

        # Assign IDs and link
        ids = []
        for idx, chunk in enumerate(deduped):
            cid = _chunk_id(chunk.text, idx)
            chunk.metadata.chunk_id    = cid
            chunk.metadata.chunk_index = idx
            ids.append(cid)

        for idx, chunk in enumerate(deduped):
            chunk.metadata.prev_chunk_id = ids[idx - 1] if idx > 0 else ""
            chunk.metadata.next_chunk_id = ids[idx + 1] if idx < len(deduped) - 1 else ""

        if self.debug:
            for c in deduped:
                logger.debug(repr(c))

        return deduped


# ─────────────────────────────────────────────
# Convenience: batch runner for multi-page docs
# ─────────────────────────────────────────────

def chunk_document(
    ocr_results:    list,
    source_file:    str   = "",
    course_id:      str   = "",
    course_name:    str   = "",
    subject_area:   str   = "",
    max_tokens:     int   = 400,
    overlap_tokens: int   = 60,
    debug:          bool  = False,
) -> list[Chunk]:
    """
    Top-level convenience function.

    Parameters
    ----------
    ocr_results  : list of OCRResult from TesseractOCREngine.run()
    source_file  : original filename (for metadata)
    course_id    : e.g. "CS201"
    course_name  : e.g. "Data Structures & Algorithms"
    subject_area : e.g. "sorting algorithms"
    max_tokens   : max tokens per prose chunk (default 400)
    overlap_tokens: overlap between prose chunks (default 60)

    Returns
    -------
    list[Chunk] — ready for vector store upsert
    """
    pipeline = ChunkingPipeline(
        source_file    = source_file,
        course_id      = course_id,
        course_name    = course_name,
        subject_area   = subject_area,
        max_tokens     = max_tokens,
        overlap_tokens = overlap_tokens,
        debug          = debug,
    )
    return pipeline.run(ocr_results)


# ─────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────

def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    """Convert chunks to plain dicts for JSON / vector store upsert."""
    return [c.to_dict() for c in chunks]


def chunks_to_pinecone_format(chunks: list[Chunk]) -> list[dict]:
    """
    Format chunks for Pinecone upsert.
    text → id, values (embedding placeholder), metadata.
    Pass `values` to your embedding model before upsert.
    """
    return [
        {
            "id":       c.metadata.chunk_id,
            "values":   [],          # Fill with embedding vector
            "metadata": {
                **c.metadata.to_dict(),
                "text": c.text,      # Stored in metadata for retrieval
            }
        }
        for c in chunks
    ]


# def chunks_to_chroma_format(chunks: list[Chunk]) -> dict:
#     """
#     Format chunks for ChromaDB collection.add().
#     Returns dict with ids, documents, metadatas.
#     """
#     return {
#         "ids":       [c.metadata.chunk_id for c in chunks],
#         "documents": [c.text for c in chunks],
#         "metadatas": [c.metadata.to_dict() for c in chunks],
#     }


# def chunks_to_weaviate_format(chunks: list[Chunk], class_name: str = "CourseChunk") -> list[dict]:
#     """Format chunks for Weaviate batch import."""
#     return [
#         {
#             "class":      class_name,
#             "id":         c.metadata.chunk_id,
#             "properties": {
#                 "text":     c.text,
#                 **c.metadata.to_dict(),
#             }
#         }
#         for c in chunks
#     ]
