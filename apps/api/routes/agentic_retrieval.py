from __future__ import annotations
import traceback
import importlib.util
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from app.core.security import get_current_user
from app.models import User

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RETRIEVAL_PIPELINE_FILE = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "retrieval pipeline"
    / "agentic_retrieval.py"
)
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_DEFAULT_TOP_K", "8"))
ASK_RESPONSE_MODEL = os.getenv(
    "GEMINI_ASK_RESPONSE_MODEL",
    os.getenv("GEMINI_FINAL_RESPONSE_MODEL", "gemini-2.5-flash"),
)
QUIZ_RESPONSE_MODEL = os.getenv(
    "GEMINI_QUIZ_RESPONSE_MODEL",
    os.getenv("GEMINI_FINAL_RESPONSE_MODEL", "gemini-2.5-pro"),
)
SUMMARY_RESPONSE_MODEL = os.getenv(
    "GEMINI_SUMMARY_RESPONSE_MODEL",
    os.getenv("GEMINI_FINAL_RESPONSE_MODEL", "gemini-2.5-pro"),
)
ASK_FALLBACK_MODEL     = os.getenv("GEMINI_ASK_FALLBACK_MODEL",     "gemini-2.5-flash-lite")
QUIZ_FALLBACK_MODEL    = os.getenv("GEMINI_QUIZ_FALLBACK_MODEL",    "gemini-2.5-flash")
SUMMARY_FALLBACK_MODEL = os.getenv("GEMINI_SUMMARY_FALLBACK_MODEL", "gemini-2.5-flash")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loader
# ─────────────────────────────────────────────────────────────────────────────

def load_retrieval_pipeline_module():
    spec = importlib.util.spec_from_file_location("agentic_retrieval_pipeline", RETRIEVAL_PIPELINE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load retrieval pipeline from {RETRIEVAL_PIPELINE_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _extract_token_count(response_obj: Any) -> int:
    """
    Extract total token count from a Gemini generate_content response.

    Gemini returns usage_metadata with:
      - prompt_token_count    (input tokens)
      - candidates_token_count (output tokens)
      - total_token_count      (sum of both)

    We use total_token_count as the billing unit for rate limiting.
    Falls back to summing input + output if total is missing.
    Returns 0 if usage_metadata is unavailable (e.g. error path).
    """
    try:
        um = getattr(response_obj, "usage_metadata", None)
        if um is None:
            return 0
        total = getattr(um, "total_token_count", None)
        if total is not None:
            return int(total)
        # Fallback: sum prompt + candidates
        prompt     = int(getattr(um, "prompt_token_count",     0) or 0)
        candidates = int(getattr(um, "candidates_token_count", 0) or 0)
        return prompt + candidates
    except Exception:
        return 0


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")
    return genai.Client(api_key=api_key.strip())


def is_retryable_gemini_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return "client has been closed" in message or "temporarily unavailable" in message


def controller_instruction(mode: Literal["ask", "quiz", "summarize"]) -> str:
    if mode == "quiz":
        return (
            "Generate the quiz requested by the student using only the retrieved course context. "
            "Include questions, answer key, explanations, and source citations like [1]."
        )
    if mode == "summarize":
        return (
            "Write a clear study summary using only the retrieved course context. "
            "Use structured headings, key terms, examples where useful, and source citations like [1]."
        )
    return (
        "Answer the student's question using only the retrieved course context. "
        "Be clear, educational, and cite factual claims with source citations like [1]."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gemini response generator — now returns (text, model, tokens_used)
# ─────────────────────────────────────────────────────────────────────────────

def generate_final_response(
    *,
    answer_prompt: str,
    mode: Literal["ask", "quiz", "summarize"],
) -> tuple[str, str, int]:
    """
    Call Gemini to generate the final student-facing response.

    Returns
    -------
    (response_text, model_used, tokens_used)
        tokens_used : total_token_count from usage_metadata (input + output).
                      This is the exact figure used for rate-limit accounting.
    """
    primary_model  = {"ask": ASK_RESPONSE_MODEL,  "quiz": QUIZ_RESPONSE_MODEL,  "summarize": SUMMARY_RESPONSE_MODEL }[mode]
    fallback_model = {"ask": ASK_FALLBACK_MODEL,   "quiz": QUIZ_FALLBACK_MODEL,  "summarize": SUMMARY_FALLBACK_MODEL  }[mode]

    config = types.GenerateContentConfig(
        system_instruction=controller_instruction(mode),
        temperature=0.2 if mode == "ask" else 0.35,
        max_output_tokens=int(os.getenv("RETRIEVAL_FINAL_MAX_OUTPUT_TOKENS", "4096")),
    )

    last_exc: Exception | None = None
    tried_models: set[str] = set()

    for model in [primary_model, fallback_model]:
        if not model or model in tried_models:
            continue
        tried_models.add(model)
        try:
            client   = get_gemini_client()
            response = client.models.generate_content(
                model    = model,
                contents = answer_prompt,
                config   = config,
            )
            tokens = _extract_token_count(response)
            logger.info(
                "Gemini response generated  model=%s  mode=%s  tokens=%d",
                model, mode, tokens,
            )
            return response.text or "", model, tokens
        except Exception as exc:
            last_exc = exc
            if not is_retryable_gemini_error(exc):
                raise
            continue

    if last_exc:
        raise last_exc
    raise RuntimeError("No Gemini response model is configured.")


# ─────────────────────────────────────────────────────────────────────────────
# Core retrieval runner — returns (payload, tokens_used)
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval(
    request: "RetrievalRequest",
    mode: Literal["ask", "quiz", "summarize"],
) -> tuple[dict, int]:
    """
    Run the full retrieval pipeline and generate the final Gemini response.

    Returns
    -------
    (payload_dict, tokens_used)
        tokens_used : total Gemini tokens for this call (0 if Gemini wasn't needed).
    """
    stage = "load retrieval pipeline"
    try:
        pipeline = load_retrieval_pipeline_module()
        stage    = "retrieve chunks"
        result   = pipeline.retrieve(
            query     = request.query,
            course_id = request.course_id,
            mode      = pipeline.RetrievalMode(mode),
            top_k     = request.top_k,
            verbose   = True,
        )
        answer_prompt = result.to_answer_prompt()
        stage         = "generate final response"

        if result.chunks:
            final_response, response_model, tokens_used = generate_final_response(
                answer_prompt=answer_prompt,
                mode=mode,
            )
        else:
            final_response = "I could not find relevant course material for this query yet."
            response_model = ""
            tokens_used    = 0

        payload = to_jsonable(result)
        if not request.include_chunks:
            payload.pop("chunks", None)
        if request.include_context:
            payload["context"] = result.to_context_string()
        if request.include_answer_prompt:
            payload["answer_prompt"] = answer_prompt
        payload["final_response"] = final_response
        payload["response_model"] = response_model
        payload["tokens_used"]    = tokens_used
        payload["pinecone_index"] = pipeline.get_course_index_name(request.course_id)

        return payload, tokens_used

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed during {stage}: {exc}",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query:                str  = Field(..., min_length=1)
    course_id:            str  = Field(..., min_length=1, description="Course identifier used to select the Pinecone index")
    top_k:                int  = Field(DEFAULT_TOP_K, ge=1, le=30)
    include_context:      bool = True
    include_answer_prompt: bool = True
    include_chunks:       bool = True


class AskRetrievalRequest(RetrievalRequest):
    pass


class QuizRetrievalRequest(RetrievalRequest):
    query: str = Field(..., min_length=1, description="Topic or learning goal for quiz generation")


class SummarizeRetrievalRequest(RetrievalRequest):
    query: str = Field(..., min_length=1, description="Topic to summarize")


router = APIRouter(prefix="/retrieval", tags=["agentic-retrieval"])


# ─────────────────────────────────────────────────────────────────────────────
# Rate-limit helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_budget(user: User, mode: str) -> "TokenRateLimitResult":
    """
    Phase 1: Pre-flight token budget check.
    Raises HTTP 429 with descriptive message if any tier is exhausted.
    """
    from rate_limiting import check_token_budget, TokenRateLimitResult

    rl = check_token_budget(
        user_id = str(user.id),
        role    = user.role.value,   # "student" | "teacher"
        mode    = mode,
    )

    if not rl.allowed:
        detail_map = {
            "sliding_window": (
                f"Token quota exceeded: you've used {rl.window_used:,} / {rl.window_limit:,} "
                f"tokens in the last 60 seconds. "
                f"Retry in {rl.retry_after_secs:.0f}s as older requests age out."
            ),
            "token_bucket": (
                f"Sending too fast: token bucket empty ({rl.bucket_tokens:.0f} remaining). "
                f"Retry in {rl.retry_after_secs:.1f}s."
            ),
            "daily_quota": (
                f"Daily token limit reached ({rl.daily_used:,} / {rl.daily_limit:,} tokens). "
                "Your quota resets at midnight UTC."
            ),
        }
        raise HTTPException(
            status_code=429,
            detail=detail_map.get(rl.rejected_by, "Token rate limit exceeded."),
            headers=rl.headers,
        )

    return rl


def _record_usage(user: User, mode: str, tokens: int) -> None:
    """
    Phase 2: Record actual token usage after Gemini responds.
    Non-fatal — errors are logged but never surface to the student.
    """
    if tokens <= 0:
        return
    from rate_limiting import record_token_usage
    record_token_usage(
        user_id       = str(user.id),
        mode          = mode,
        actual_tokens = tokens,
        role          = user.role.value,
    )


def _attach_headers(response: Response, rl: "TokenRateLimitResult", tokens_used: int) -> None:
    """Inject rate-limit headers + token cost on every successful response."""
    for key, value in rl.headers.items():
        response.headers[key] = value
    response.headers["X-Tokens-Used-This-Request"] = str(tokens_used)


# ─────────────────────────────────────────────────────────────────────────────
# Route endpoints — authenticated + token-rate-limited
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/ask")
async def ask(
    request:  AskRetrievalRequest,
    response: Response,
    user:     User = Depends(get_current_user),
) -> dict:
    # Phase 1: check budget BEFORE spending any tokens
    rl = _check_budget(user, "ask")

    # Run retrieval + Gemini (returns actual token count)
    payload, tokens_used = await run_in_threadpool(run_retrieval, request, "ask")

    # Phase 2: record actual cost AFTER Gemini responds
    await run_in_threadpool(_record_usage, user, "ask", tokens_used)

    _attach_headers(response, rl, tokens_used)
    payload["rate_limit"] = {
        "window_tokens_remaining": rl.window_remaining,
        "daily_tokens_remaining":  rl.daily_remaining,
        "bucket_tokens":           int(rl.bucket_tokens),
        "tokens_used_this_call":   tokens_used,
    }
    return payload


@router.post("/quiz")
async def quiz(
    request:  QuizRetrievalRequest,
    response: Response,
    user:     User = Depends(get_current_user),
) -> dict:
    rl = _check_budget(user, "quiz")
    payload, tokens_used = await run_in_threadpool(run_retrieval, request, "quiz")
    await run_in_threadpool(_record_usage, user, "quiz", tokens_used)
    _attach_headers(response, rl, tokens_used)
    payload["rate_limit"] = {
        "window_tokens_remaining": rl.window_remaining,
        "daily_tokens_remaining":  rl.daily_remaining,
        "bucket_tokens":           int(rl.bucket_tokens),
        "tokens_used_this_call":   tokens_used,
    }
    return payload


@router.post("/summarize")
async def summarize(
    request:  SummarizeRetrievalRequest,
    response: Response,
    user:     User = Depends(get_current_user),
) -> dict:
    rl = _check_budget(user, "summarize")
    payload, tokens_used = await run_in_threadpool(run_retrieval, request, "summarize")
    await run_in_threadpool(_record_usage, user, "summarize", tokens_used)
    _attach_headers(response, rl, tokens_used)
    payload["rate_limit"] = {
        "window_tokens_remaining": rl.window_remaining,
        "daily_tokens_remaining":  rl.daily_remaining,
        "bucket_tokens":           int(rl.bucket_tokens),
        "tokens_used_this_call":   tokens_used,
    }
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Admin endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/rate-limit/me")
async def my_token_usage(user: User = Depends(get_current_user)) -> dict:
    """Show current token usage state for the authenticated user."""
    from rate_limiting import get_user_usage
    return get_user_usage(str(user.id))


@router.delete("/rate-limit/{user_id}/reset")
async def reset_rate_limit(
    user_id: str,
    user:    User = Depends(get_current_user),
) -> dict:
    """Teacher-only: reset all token rate-limit state for a specific user."""
    if user.role.value != "teacher":
        raise HTTPException(403, "Only teachers can reset token rate limits.")
    from rate_limiting import reset_user_limits
    deleted = reset_user_limits(user_id)
    return {"reset": True, "user_id": user_id, "redis_keys_deleted": deleted}
