from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


CURRENT_DIR = Path(__file__).resolve().parent
PROMPT_FILE = CURRENT_DIR / "prompts" / "transcription.md"
TRANSCRIPTION_MODEL = os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-2.5-flash")

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


class TranscriptSegment(BaseModel):
    segment_index: int = Field(..., ge=0)
    start_seconds: float = Field(..., ge=0.0)
    end_seconds: float = Field(..., ge=0.0)
    speaker: str = Field(..., description="Stable label such as Speaker 1")
    language: str = Field(..., description="Detected spoken language")
    text: str = Field(..., description="Faithful segment transcription")
    emotion: str = Field("unknown", description="Detected emotion when clear")
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_unclear: bool = False
    notes: str = ""


class TranscriptMetadata(BaseModel):
    source_file: str
    file_name: str
    file_type: str
    mime_type: str
    model: str
    duration_seconds: Optional[float] = Field(None, ge=0.0)
    detected_languages: List[str] = Field(default_factory=list)
    speaker_count: int = Field(..., ge=0)
    speakers: List[str] = Field(default_factory=list)
    segment_count: int = Field(..., ge=0)
    transcript: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    summary: str
    topics: List[str] = Field(default_factory=list)
    searchable_keywords: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class GeminiTranscriptResponse(BaseModel):
    duration_seconds: Optional[float] = Field(None, ge=0.0)
    detected_languages: List[str] = Field(default_factory=list)
    speaker_count: int = Field(..., ge=0)
    speakers: List[str] = Field(default_factory=list)
    transcript: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    summary: str
    topics: List[str] = Field(default_factory=list)
    searchable_keywords: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def _get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")
    return genai.Client(api_key=api_key)


def _resolve_audio_path(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            "Unsupported audio type. Supported: "
            + ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        )
    return path


def _detect_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "audio/mpeg"


def _load_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def generate_transcript(file_path: str | Path) -> TranscriptMetadata:
    """Generate a structured transcript from a local audio file."""
    path = _resolve_audio_path(file_path)
    mime_type = _detect_mime_type(path)
    client = _get_gemini_client()

    response = client.models.generate_content(
        model=TRANSCRIPTION_MODEL,
        contents=[
            _load_prompt(),
            types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiTranscriptResponse,
            temperature=0.1,
        ),
    )

    parsed = getattr(response, "parsed", None)
    transcript = (
        parsed
        if isinstance(parsed, GeminiTranscriptResponse)
        else GeminiTranscriptResponse.model_validate_json(response.text)
    )

    return TranscriptMetadata(
        source_file=str(path),
        file_name=path.name,
        file_type=path.suffix.lower().lstrip("."),
        mime_type=mime_type,
        model=TRANSCRIPTION_MODEL,
        duration_seconds=transcript.duration_seconds,
        detected_languages=transcript.detected_languages,
        speaker_count=transcript.speaker_count,
        speakers=transcript.speakers,
        segment_count=len(transcript.segments),
        transcript=transcript.transcript,
        segments=transcript.segments,
        summary=transcript.summary,
        topics=transcript.topics,
        searchable_keywords=transcript.searchable_keywords,
        warnings=transcript.warnings,
    )


def generate_transcript_dict(file_path: str | Path) -> dict:
    """JSON-ready wrapper for workers, routes, and tests."""
    return generate_transcript(file_path).model_dump()
