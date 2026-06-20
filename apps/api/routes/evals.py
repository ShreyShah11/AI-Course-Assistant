"""
Local-only evaluation routes.
==============================
Shown in Swagger under "Eval Pipeline" tag.
Intended for localhost use only (Postman, curl, Swagger UI).

Endpoints
---------
  POST  /internal/evals/run              Synthetic eval — fixtures only, instant
  POST  /internal/evals/run-live         LIVE eval — real Pinecone + Gemini
  GET   /internal/evals/last-report      Last synthetic report (cached)
  GET   /internal/evals/last-live-report Last live report (cached)

Live eval usage
---------------
  # Run all 3 test-suite scenarios against your real Pinecone index
  curl -s -X POST http://localhost:8000/internal/evals/run-live \\
       -H "Content-Type: application/json" \\
       -d '{"course_id": "CS301"}' | python -m json.tool

  # With LLM judge
  curl -s -X POST http://localhost:8000/internal/evals/run-live \\
       -H "Content-Type: application/json" \\
       -d '{"course_id": "CS301", "include_llm_judge": true}' | python -m json.tool
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/evals", tags=["Eval Pipeline"])

_last_synthetic: Optional[dict[str, Any]] = None
_last_live:      Optional[dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class EvalRunRequest(BaseModel):
    include_llm_judge: bool = Field(
        default=True,
        description="Call Gemini LLM judge if EVAL_LLM_JUDGE_MODEL is set in .env.",
    )


class LiveEvalRunRequest(BaseModel):
    course_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Pinecone index name to seed and query. "
            "Created automatically if it doesn't exist. Example: 'Test01'"
        ),
    )
    include_llm_judge: bool = Field(
        default=False,
        description=(
            "When True, calls Gemini (EVAL_LLM_JUDGE_MODEL) to score faithfulness "
            "for each scenario. Adds ~5-15 s per scenario."
        ),
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Chunks to retrieve per query from Pinecone.",
    )
    seed_first: bool = Field(
        default=True,
        description=(
            "When True (default), embeds all fixture chunks from tests.py and "
            "upserts them into the Pinecone index before running retrieval. "
            "Set to False to skip seeding and reuse a previously seeded index "
            "(saves ~2-3 minutes on repeat runs)."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/run", summary="Synthetic eval — fixtures only, ~1 second")
def run_evals(body: EvalRunRequest = EvalRunRequest()) -> dict[str, Any]:
    """
    Run the **synthetic** evaluation suite (no Pinecone, no live calls).

    Uses pre-built fixtures from `evals/Test suites/tests.py` to check:
    - Chunking quality (size, metadata completeness)
    - Retrieval metrics (MRR, NDCG, Hit Rate) against known ground truth
    - Faithfulness metrics (hallucination proxy, keyword recall)
    - Latency SLO compliance (simulated timing from fixtures)

    Completes in ~1-2 seconds. Use for quick pipeline design checks.
    Use **POST /internal/evals/run-live** for real Pinecone evaluation.
    """
    global _last_synthetic
    logger.info("POST /internal/evals/run  include_llm_judge=%s", body.include_llm_judge)
    try:
        from evals.runner import run_eval_suite
        report = run_eval_suite(include_llm_judge=body.include_llm_judge)
        _last_synthetic = report
        h = report.get("overall_health", {})
        logger.info(
            "Synthetic eval done  pass=%s  mrr=%.3f  hallucination=%.3f",
            h.get("pass"), h.get("avg_mrr", 0), h.get("avg_hallucination_proxy", 0),
        )
        return report
    except Exception as exc:
        logger.exception("Synthetic eval failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Synthetic eval error: {exc}") from exc


@router.post("/run-live", summary="LIVE eval — real Pinecone + Gemini, ~30-60 s")
async def run_live_evals(body: LiveEvalRunRequest) -> dict[str, Any]:
    """
    Run the **live** end-to-end evaluation using test-suite scenarios against
    your real Pinecone index.

    For each scenario in `evals/Test suites/tests.py`:
    - Calls `pipeline.retrieve()` → real Pinecone dense search + BM25 + MMR + RRF
    - Generates a real Gemini answer (same code as `/retrieval/ask`)
    - Measures wall-clock latency end-to-end
    - Evaluates **chunking quality** of real returned chunks
      (metadata completeness, size distribution, text richness)
    - Evaluates **retrieval quality** against ground-truth `relevant_chunk_ids`
      from tests.py: MRR, NDCG@5, NDCG@10, Precision@5, Hit-Rate@5,
      source coverage (expected namespaces present?), source diversity
    - Evaluates **faithfulness** of the real Gemini answer vs real context:
      hallucination proxy, keyword recall, citation density
    - Optionally scores with Gemini LLM-as-judge (`include_llm_judge: true`)

    **Requires:** `course_id` must have data in Pinecone.
    **Takes:** ~10-60 seconds depending on Pinecone latency and LLM judge.
    """
    global _last_live
    logger.info(
        "POST /internal/evals/run-live  course_id=%s  judge=%s  top_k=%d",
        body.course_id, body.include_llm_judge, body.top_k,
    )
    try:
        from evals.runner_live import run_live_eval
        report = await run_in_threadpool(
            run_live_eval,
            body.course_id,
            None,                    # queries — always from test suites
            body.include_llm_judge,
            body.top_k,
            body.seed_first,
        )
        _last_live = report
        h = report.get("overall_health", {})
        logger.info(
            "Live eval done  course_id=%s  pass=%s  mrr=%.3f  ndcg5=%.3f  "
            "latency_ms=%.0f  hallucination=%.3f",
            body.course_id,
            h.get("pass_eval"),
            h.get("avg_mrr", 0),
            h.get("avg_ndcg_at_5", 0),
            h.get("avg_latency_ms", 0),
            h.get("avg_hallucination_proxy", 0),
        )
        return report
    except Exception as exc:
        logger.exception("Live eval failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Live eval error: {exc}") from exc


@router.get("/last-report", summary="Last synthetic eval report")
def last_report() -> dict[str, Any]:
    """Return the most recent synthetic eval report cached in memory."""
    if _last_synthetic is None:
        raise HTTPException(
            status_code=404,
            detail="No synthetic report yet. Call POST /internal/evals/run first.",
        )
    return _last_synthetic


@router.get("/last-live-report", summary="Last live eval report")
def last_live_report() -> dict[str, Any]:
    """Return the most recent live eval report cached in memory."""
    if _last_live is None:
        raise HTTPException(
            status_code=404,
            detail="No live report yet. Call POST /internal/evals/run-live first.",
        )
    return _last_live
