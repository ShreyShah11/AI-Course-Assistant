"""
Image ingestion chunking pipeline.

This module receives page-wise OCR results from the image OCR worker and turns
them into clean, LLM-ready chunks. It does not extract semantic metadata locally.
Gemini enrichment in the worker handles summary, topic, keywords, and content
signals after these chunks are created.
"""

from __future__ import annotations

import hashlib
import logging
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    chunk_id: str = ""
    chunk_index: int = 0
    chunk_type: str = "prose"

    source_file: str = ""
    document_id: str = ""
    page_numbers: list[int] = field(default_factory=list)
    total_pages: int = 0

    course_id: str = ""
    course_name: str = ""
    subject_area: str = ""
    document_title: str = ""
    chapter: str = ""
    section: str = ""
    subsection: str = ""

    heading_path: list[str] = field(default_factory=list)
    prev_chunk_id: str = ""
    next_chunk_id: str = ""
    is_continuation: bool = False
    part_index: int = 0

    token_count: int = 0
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    language: str = ""
    content_signals: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    ocr_confidence: float = 0.0
    image_type: str = ""
    skew_corrected: bool = False
    skew_angle: float = 0.0
    ocr_warnings: list[str] = field(default_factory=list)
    preprocessing_stages: list[str] = field(default_factory=list)

    has_code: bool = False
    has_formula: bool = False
    has_list: bool = False
    is_definition: bool = False
    is_example: bool = False
    semantic_density: float = 0.0
    context_window: str = ""

    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata.to_dict()}


@dataclass
class _TextUnit:
    text: str
    page: int
    section: str
    confidence: float
    image_type: str
    skew_corrected: bool
    skew_angle: float
    warnings: list[str]
    stages: list[str]


def _chunk_id(text: str, index: int) -> str:
    payload = f"{index}:{text[:200]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _doc_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ChunkingPipeline:
    """
    Section-aware text chunker for OCR output.

    Responsibilities:
    - normalize OCR text
    - preserve page and OCR quality metadata
    - pack text into embedding-friendly windows with overlap
    - link chunks with stable ids
    """

    def __init__(
        self,
        source_file: str = "",
        course_id: str = "",
        course_name: str = "",
        subject_area: str = "",
        max_tokens: int = 420,
        overlap_tokens: int = 70,
        min_chunk_tokens: int = 25,
        merge_headings: bool = True,
        debug: bool = False,
    ):
        self.source_file = source_file
        self.course_id = course_id
        self.course_name = course_name
        self.subject_area = subject_area
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.merge_headings = merge_headings
        self.debug = debug

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _page_text(cls, ocr_result) -> str:
        blocks = getattr(ocr_result, "blocks", None) or []
        if blocks:
            block_texts = [getattr(block, "text", "").strip() for block in blocks]
            text = "\n\n".join(text for text in block_texts if text)
            if text:
                return cls._normalize_text(text)
        return cls._normalize_text(getattr(ocr_result, "full_text", ""))

    @classmethod
    def _split_paragraphs(cls, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        merged: list[str] = []
        buffer: list[str] = []
        for line in lines:
            buffer.append(line)
            if re.search(r"[.!?;:]$", line) or len(" ".join(buffer).split()) >= 90:
                merged.append(" ".join(buffer).strip())
                buffer = []
        if buffer:
            merged.append(" ".join(buffer).strip())

        return merged or ([text.strip()] if text.strip() else [])

    @staticmethod
    def _looks_like_section(text: str) -> bool:
        compact = " ".join(text.split())
        if not compact or len(compact) > 110 or compact.endswith((".", ",", ";")):
            return False
        if re.match(r"^(\d+(\.\d+)*|chapter|unit|module|lesson|slide)\b", compact, re.I):
            return True
        words = compact.split()
        return len(words) <= 10 and (compact.isupper() or compact.istitle())

    @classmethod
    def _first_title(cls, text: str) -> str:
        for paragraph in cls._split_paragraphs(text):
            title = " ".join(paragraph.split())
            if title:
                return title[:120]
        return ""

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                output.append(value)
        return output

    def _units_from_pages(self, ocr_results: list) -> tuple[list[_TextUnit], str, str]:
        page_texts = [self._page_text(result) for result in ocr_results]
        full_text = "\n\n".join(text for text in page_texts if text)
        document_title = self._first_title(full_text)
        current_section = document_title
        units: list[_TextUnit] = []

        for ocr_result, page_text in zip(ocr_results, page_texts):
            if not page_text:
                continue

            for paragraph in self._split_paragraphs(page_text):
                if self._looks_like_section(paragraph):
                    current_section = " ".join(paragraph.split())[:160]
                    if self.merge_headings:
                        continue

                text = paragraph
                if self.merge_headings and current_section and current_section not in text[:180]:
                    text = f"{current_section}\n\n{text}"

                units.append(
                    _TextUnit(
                        text=text,
                        page=getattr(ocr_result, "page", 1),
                        section=current_section if current_section != document_title else "",
                        confidence=float(getattr(ocr_result, "mean_confidence", 0.0) or 0.0),
                        image_type=getattr(ocr_result, "image_type", ""),
                        skew_corrected=bool(getattr(ocr_result, "skew_corrected", False)),
                        skew_angle=float(getattr(ocr_result, "skew_angle", 0.0) or 0.0),
                        warnings=list(getattr(ocr_result, "warnings", []) or []),
                        stages=list(getattr(ocr_result, "stages_applied", []) or []),
                    )
                )

        return units, full_text, document_title

    def _split_large_text(self, text: str) -> list[str]:
        if self._approx_tokens(text) <= self.max_tokens:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            words = text.split()
            window_words = max(80, int(self.max_tokens / 1.3))
            overlap_words = max(0, int(self.overlap_tokens / 1.3))
            step = max(1, window_words - overlap_words)
            return [" ".join(words[i : i + window_words]).strip() for i in range(0, len(words), step)]

        parts: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            candidate = " ".join([*current, sentence]).strip()
            if current and self._approx_tokens(candidate) > self.max_tokens:
                parts.append(" ".join(current).strip())
                overlap_words = max(1, int(self.overlap_tokens / 1.3))
                overlap = " ".join(parts[-1].split()[-overlap_words:])
                current = [overlap, sentence] if overlap else [sentence]
            else:
                current.append(sentence)
        if current:
            parts.append(" ".join(current).strip())

        return [part for part in parts if part]

    def _overlap_tail(self, window: list[_TextUnit], next_unit: _TextUnit) -> _TextUnit | None:
        if self.overlap_tokens <= 0:
            return None
        words = " ".join(unit.text for unit in window).split()
        if not words:
            return None

        overlap_words = max(1, int(self.overlap_tokens / 1.3))
        return _TextUnit(
            text=" ".join(words[-overlap_words:]),
            page=next_unit.page,
            section=next_unit.section,
            confidence=next_unit.confidence,
            image_type=next_unit.image_type,
            skew_corrected=next_unit.skew_corrected,
            skew_angle=next_unit.skew_angle,
            warnings=[],
            stages=[],
        )

    def _pack_units(self, units: list[_TextUnit]) -> list[list[_TextUnit]]:
        windows: list[list[_TextUnit]] = []
        current: list[_TextUnit] = []
        current_tokens = 0

        for unit in units:
            for part in self._split_large_text(unit.text):
                part_unit = deepcopy(unit)
                part_unit.text = part
                part_tokens = self._approx_tokens(part)

                if current and current_tokens + part_tokens > self.max_tokens:
                    windows.append(current)
                    overlap = self._overlap_tail(current, part_unit)
                    current = [overlap] if overlap else []
                    current_tokens = self._approx_tokens(overlap.text) if overlap else 0

                current.append(part_unit)
                current_tokens += part_tokens

        if current:
            windows.append(current)

        return self._merge_tiny_windows(windows)

    @staticmethod
    def _window_text(window: list[_TextUnit]) -> str:
        parts: list[str] = []
        for unit in window:
            text = unit.text.strip()
            if text and (not parts or parts[-1] != text):
                parts.append(text)
        return "\n\n".join(parts).strip()

    def _merge_tiny_windows(self, windows: list[list[_TextUnit]]) -> list[list[_TextUnit]]:
        merged: list[list[_TextUnit]] = []
        for window in windows:
            token_count = self._approx_tokens(self._window_text(window))
            if merged and token_count < self.min_chunk_tokens * 2:
                candidate = [*merged[-1], *window]
                if self._approx_tokens(self._window_text(candidate)) <= self.max_tokens + self.overlap_tokens:
                    merged[-1] = candidate
                    continue
            merged.append(window)
        return merged

    def _metadata_for_window(
        self,
        window: list[_TextUnit],
        text: str,
        doc_id: str,
        total_pages: int,
        document_title: str,
        part_index: int,
    ) -> ChunkMetadata:
        words = text.split()
        word_count = len(words)
        unique_words = {word.lower() for word in words if word.isalpha()}
        confidences = [unit.confidence for unit in window if unit.confidence > 0]
        page_numbers = sorted({unit.page for unit in window})
        section = next((unit.section for unit in window if unit.section), "")

        return ChunkMetadata(
            source_file=self.source_file,
            document_id=doc_id,
            page_numbers=page_numbers,
            total_pages=total_pages,
            course_id=self.course_id,
            course_name=self.course_name,
            subject_area=self.subject_area,
            document_title=document_title,
            section=section,
            heading_path=[section] if section else [],
            is_continuation=part_index > 0,
            part_index=part_index,
            token_count=self._approx_tokens(text),
            char_count=len(text),
            word_count=word_count,
            line_count=text.count("\n") + 1,
            ocr_confidence=round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
            image_type=next((unit.image_type for unit in window if unit.image_type), ""),
            skew_corrected=any(unit.skew_corrected for unit in window),
            skew_angle=round(sum(unit.skew_angle for unit in window) / max(1, len(window)), 2),
            ocr_warnings=self._unique([warning for unit in window for warning in unit.warnings])[:20],
            preprocessing_stages=self._unique([stage for unit in window for stage in unit.stages])[:30],
            semantic_density=round(len(unique_words) / max(1, word_count), 3),
            context_window=text[:500],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _post_process(self, chunks: list[Chunk]) -> list[Chunk]:
        filtered = [chunk for chunk in chunks if self._approx_tokens(chunk.text) >= self.min_chunk_tokens]
        deduped: list[Chunk] = []
        seen: set[str] = set()

        for chunk in filtered:
            digest = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()[:16]
            if digest in seen:
                continue
            seen.add(digest)
            deduped.append(chunk)

        ids: list[str] = []
        for index, chunk in enumerate(deduped):
            chunk_id = _chunk_id(chunk.text, index)
            chunk.metadata.chunk_id = chunk_id
            chunk.metadata.chunk_index = index
            ids.append(chunk_id)

        for index, chunk in enumerate(deduped):
            chunk.metadata.prev_chunk_id = ids[index - 1] if index > 0 else ""
            chunk.metadata.next_chunk_id = ids[index + 1] if index < len(deduped) - 1 else ""

        return deduped

    def run(self, ocr_results: list) -> list[Chunk]:
        if not ocr_results:
            return []

        units, full_text, document_title = self._units_from_pages(ocr_results)
        if not units:
            return []

        doc_id = _doc_id(full_text)
        total_pages = len(ocr_results)
        chunks: list[Chunk] = []

        for index, window in enumerate(self._pack_units(units)):
            text = self._window_text(window)
            if not text:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    metadata=self._metadata_for_window(
                        window=window,
                        text=text,
                        doc_id=doc_id,
                        total_pages=total_pages,
                        document_title=document_title,
                        part_index=index,
                    ),
                )
            )

        output = self._post_process(chunks)
        logger.info(
            "Image chunking complete: %d chunks from %d page(s) of %s",
            len(output),
            total_pages,
            self.source_file,
        )
        return output


def chunk_document(
    ocr_results: list,
    source_file: str = "",
    course_id: str = "",
    course_name: str = "",
    subject_area: str = "",
    max_tokens: int = 400,
    overlap_tokens: int = 60,
    debug: bool = False,
) -> list[Chunk]:
    pipeline = ChunkingPipeline(
        source_file=source_file,
        course_id=course_id,
        course_name=course_name,
        subject_area=subject_area,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        debug=debug,
    )
    return pipeline.run(ocr_results)


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [chunk.to_dict() for chunk in chunks]


def chunks_to_pinecone_format(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "id": chunk.metadata.chunk_id,
            "values": [],
            "metadata": {**chunk.metadata.to_dict(), "text": chunk.text},
        }
        for chunk in chunks
    ]
