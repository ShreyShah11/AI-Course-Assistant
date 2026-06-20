"""
Eval Suite Runner
=================
Orchestrates all five metric modules against the synthetic, deterministic
fixtures in 'evals/Test suites/tests.py'.  No live Pinecone or LLM calls
are needed — the fixtures are self-contained.

Usage
-----
    from evals.runner import run_eval_suite

    report = run_eval_suite()                      # uses env for LLM judge
    report = run_eval_suite(include_llm_judge=False)  # skip LLM judge

Returned EvalReport shape
-------------------------
{
  "run_id":         str,
  "run_at":         ISO-8601 str,
  "llm_judge_used": bool,
  "llm_judge_model": str | None,
  "chunking":       { "qna": {...}, "audio": {...}, ... },
  "retrieval":      [ { "scenario_id": ..., "MRR": ..., ... } ],
  "faithfulness":   [ { "scenario_id": ..., "hallucination_proxy": ..., "llm_judge_score": ... } ],
  "coverage":       { "source_namespace_coverage": {...}, ... },
  "latency":        { "total": {...}, "planner": {...} },
  "overall_health": { "pass": bool, ... }
}
"""

from __future__ import annotations

import importlib.util
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from evals.chunking_metrics import ChunkingMetrics
from evals.retrieval_metrics import RetrievalMetrics
from evals.faithfulness import FaithfulnessMetrics
from evals.coverage import CoverageMetrics
from evals.latency import LatencyMetrics, LatencyReport
from evals.llm_judge import LLMJudge

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Health thresholds
# Tune these upward as the pipeline matures.
# ─────────────────────────────────────────────────────────────────────────────

HEALTH_THRESHOLDS: dict[str, float] = {
    # All chunking pipelines must have ≥ this metadata completeness
    "metadata_completeness_min":  0.85,
    # Average MRR across retrieval scenarios
    "mrr_min":                    0.40,
    # Average hit rate across retrieval scenarios
    "hit_rate_min":               0.50,
    # Average hallucination proxy (lower is better)
    "hallucination_proxy_max":    0.35,
    # Average keyword recall
    "keyword_recall_min":         0.50,
    # Fraction of syllabus topics covered
    "topic_coverage_min":         0.30,
    # LLM judge overall score (when judge is active)
    "llm_judge_score_min":        0.60,
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixture loader
# (The directory name "Test suites" has a space — can't use normal dot-import)
# ─────────────────────────────────────────────────────────────────────────────

def _load_fixtures():
    """Dynamically load evals/Test suites/tests.py and return the module."""
    tests_path = Path(__file__).parent / "Test suites" / "tests.py"
    spec   = importlib.util.spec_from_file_location("evals_tests", tests_path)
    module = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(module)                  # type: ignore[union-attr]
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_suite(include_llm_judge: bool = True) -> dict[str, Any]:
    """
    Run the full evaluation suite and return a structured EvalReport.

    Parameters
    ----------
    include_llm_judge : bool
        When True (default) and EVAL_LLM_JUDGE_MODEL is set in .env, calls
        Gemini to score faithfulness for each scenario's sample answer.
        When False, the LLM judge is always skipped.
    """
    logger.info("Starting eval suite run …")
    fixtures = _load_fixtures()

    run_id = str(uuid.uuid4())
    run_at = datetime.now(timezone.utc).isoformat()
    use_llm = include_llm_judge and LLMJudge.is_available()

    if use_llm:
        logger.info("LLM judge active: model=%s", os.getenv("EVAL_LLM_JUDGE_MODEL"))
    else:
        logger.info("LLM judge inactive (EVAL_LLM_JUDGE_MODEL not set or include_llm_judge=False)")

    # ── 1. Chunking metrics ───────────────────────────────────────────────────
    logger.info("Running chunking metrics …")
    chunking_results: dict[str, Any] = {
        "qna": ChunkingMetrics.evaluate_qna(fixtures.QNA_CHUNKS).summary(),
        "audio": ChunkingMetrics.evaluate_audio(
            fixtures.AUDIO_CHUNKS,
            lecture_duration_sec=3240.0,   # from the lecture_summary fixture
        ).summary(),
        "documents": ChunkingMetrics.evaluate_documents(fixtures.DOCUMENT_CHUNKS).summary(),
        "image":     ChunkingMetrics.evaluate_image(fixtures.IMAGE_CHUNKS).summary(),
        "handwritten": ChunkingMetrics.evaluate_handwritten(fixtures.HANDWRITTEN_CHUNKS).summary(),
        "youtube": ChunkingMetrics.evaluate_youtube(
            fixtures.YOUTUBE_CHUNKS,
            video_chapters=[
                {"title": "Introduction",            "start_sec": 0},
                {"title": "Worked Example: Fibonacci", "start_sec": 540},
            ],
        ).summary(),
    }

    # ── 2. Retrieval metrics ──────────────────────────────────────────────────
    logger.info("Running retrieval metrics …")
    retrieval_results: list[dict[str, Any]] = []

    for scenario in fixtures.RETRIEVAL_SCENARIOS:
        r = RetrievalMetrics.evaluate(
            retrieved=scenario["retrieved_chunks"],
            relevant_ids=scenario["relevant_chunk_ids"],
            expected_sources=scenario["expected_sources"],
            k=scenario.get("k", 8),
            relevance_grades=scenario.get("relevance_grades"),
            original_sub_query_count=scenario.get("original_sub_query_count", 0),
            deduplicated_sub_query_count=scenario.get("deduplicated_sub_query_count", 0),
            planner_ms=scenario.get("planner_ms", 0.0),
            total_ms=scenario.get("total_ms", 0.0),
        )
        retrieval_results.append({
            "scenario_id": scenario["id"],
            "query":       scenario["query"],
            "mode":        scenario["mode"],
            **r.summary(),
        })

    # ── 3. Faithfulness metrics ───────────────────────────────────────────────
    logger.info("Running faithfulness metrics …")
    faithfulness_results: list[dict[str, Any]] = []

    for scenario in fixtures.RETRIEVAL_SCENARIOS:
        answer = scenario.get("sample_answer", "")
        if not answer:
            continue

        plan    = scenario.get("planner_data", {})
        chunks  = scenario["retrieved_chunks"]

        # Build a rich context string from every useful text field on each chunk.
        # In the test fixtures the display_text is a short placeholder — we therefore
        # assemble context from all available text-bearing fields so the faithfulness
        # heuristics (keyword overlap, hallucination proxy) have meaningful signal.
        context_parts: list[str] = []
        for chunk in chunks:
            parts: list[str] = []
            display = chunk.get("display_text", "")
            if display:
                parts.append(display)
            for field in ("topic", "section_title", "lecture_title",
                          "chapter_title", "image_summary"):
                val = chunk.get(field, "")
                if val:
                    parts.append(str(val))
            kws = chunk.get("keywords", [])
            if isinstance(kws, list) and kws:
                parts.append(", ".join(kws))
            context_parts.append(" | ".join(filter(None, parts)))


        # Also include domain_keywords and key_concepts from the planner so the
        # keyword_recall metric has something to match against.
        plan_text_parts: list[str] = []
        plan_text_parts.extend(plan.get("domain_keywords", []))
        plan_text_parts.extend(plan.get("key_concepts", []))
        if plan_text_parts:
            context_parts.append(" ".join(plan_text_parts))

        context = "\n\n".join(filter(None, context_parts))


        f = FaithfulnessMetrics.evaluate(
            answer=answer,
            context_string=context,
            num_chunks=len(chunks),
            domain_keywords=plan.get("domain_keywords", []),
            expected_answer_length=plan.get("expected_answer_length", "medium"),
            expected_answer_format=plan.get("expected_answer_format", "structured_explanation"),
        )

        entry: dict[str, Any] = {
            "scenario_id":         scenario["id"],
            "query":               scenario["query"],
            "mode":                scenario["mode"],
            **f.summary(),
            "llm_judge_score":     None,
            "llm_judge_reasoning": None,
            "llm_judge_model":     None,
        }

        if use_llm:
            judge = LLMJudge.score(
                question=scenario["query"],
                context=context,
                answer=answer,
            )
            if judge is not None:
                entry["llm_judge_score"]     = round(judge.score, 4)
                entry["llm_judge_reasoning"] = judge.reasoning
                entry["llm_judge_model"]     = judge.model_used
                logger.info(
                    "LLM judge [%s]: score=%.3f  %s",
                    scenario["id"], judge.score, judge.reasoning[:80]
                )

        faithfulness_results.append(entry)

    # ── 4. Coverage metrics ───────────────────────────────────────────────────
    logger.info("Running coverage metrics …")
    suite_for_coverage = [
        {
            "mode":             s["mode"],
            "chunks":           s["retrieved_chunks"],
            "plan":             s.get("planner_data", {}),
            "expected_sources": list(s.get("expected_sources", [])),
            "excluded_sources": list(s.get("excluded_sources", [])),
        }
        for s in fixtures.RETRIEVAL_SCENARIOS
    ]
    coverage_result = CoverageMetrics.evaluate_suite(
        suite_results=suite_for_coverage,
        syllabus_topics=fixtures.SYLLABUS_TOPICS,
    )

    # Mode adaptation score (compares source weights across ask / quiz / summarize)
    ask_w  = _extract_source_weights(fixtures.RETRIEVAL_SCENARIOS, "ask")
    quiz_w = _extract_source_weights(fixtures.RETRIEVAL_SCENARIOS, "quiz")
    sum_w  = _extract_source_weights(fixtures.RETRIEVAL_SCENARIOS, "summarize")
    mode_adapt = CoverageMetrics.mode_adaptation_score(ask_w, quiz_w, sum_w)

    coverage_summary = coverage_result.summary()
    coverage_summary["mode_adaptation_score"] = round(mode_adapt, 4)

    # ── 5. Latency metrics ────────────────────────────────────────────────────
    logger.info("Running latency metrics …")
    lat_report = LatencyReport()
    for s in fixtures.RETRIEVAL_SCENARIOS:
        if s.get("total_ms"):
            lat_report.add("total",   float(s["total_ms"]))
        if s.get("planner_ms"):
            lat_report.add("planner", float(s["planner_ms"]))
    lat_report.compute()

    # ── 6. Overall health ─────────────────────────────────────────────────────
    health = _compute_health(
        chunking_results, retrieval_results, faithfulness_results,
        coverage_summary, lat_report
    )

    logger.info(
        "Eval suite complete. run_id=%s  pass=%s  mrr=%.3f  hallucination_proxy=%.3f",
        run_id,
        health["pass"],
        health.get("avg_mrr", 0),
        health.get("avg_hallucination_proxy", 0),
    )

    return {
        "run_id":          run_id,
        "run_at":          run_at,
        "llm_judge_used":  use_llm,
        "llm_judge_model": os.getenv("EVAL_LLM_JUDGE_MODEL", "").strip() if use_llm else None,
        "chunking":        chunking_results,
        "retrieval":       retrieval_results,
        "faithfulness":    faithfulness_results,
        "coverage":        coverage_summary,
        "latency":         lat_report.summary(),
        "overall_health":  health,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_source_weights(scenarios: list[dict], mode: str) -> dict[str, float]:
    """
    Average priority_weight per source namespace across all scenarios of
    the given mode (ask | quiz | summarize).
    """
    totals: dict[str, list[float]] = {}
    for s in scenarios:
        if s.get("mode") != mode:
            continue
        for budget in s.get("planner_data", {}).get("source_budgets", []):
            sk = budget.get("source_key", "")
            wt = float(budget.get("priority_weight", 1.0))
            totals.setdefault(sk, []).append(wt)
    return {k: sum(v) / len(v) for k, v in totals.items()}


def _compute_health(
    chunking:     dict[str, Any],
    retrieval:    list[dict[str, Any]],
    faithfulness: list[dict[str, Any]],
    coverage:     dict[str, Any],
    lat_report:   LatencyReport,
) -> dict[str, Any]:
    """
    Compute top-level pass/fail health signals using HEALTH_THRESHOLDS.
    """
    t = HEALTH_THRESHOLDS

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunking_ok = all(
        p.get("metadata_completeness", {}).get("overall_completeness", 0.0)
        >= t["metadata_completeness_min"]
        for p in chunking.values()
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    mrr_vals  = [r.get("MRR", 0.0) for r in retrieval]
    hit_vals  = [r.get("Hit Rate", 0.0) for r in retrieval]
    avg_mrr      = sum(mrr_vals) / len(mrr_vals)   if mrr_vals  else 0.0
    avg_hit_rate = sum(hit_vals) / len(hit_vals)   if hit_vals  else 0.0
    retrieval_ok = avg_mrr >= t["mrr_min"] and avg_hit_rate >= t["hit_rate_min"]

    # ── Faithfulness ─────────────────────────────────────────────────────────
    hall_vals  = [f.get("hallucination_proxy", 1.0) for f in faithfulness]
    kw_vals    = [f.get("keyword_recall", 0.0)      for f in faithfulness]
    judge_vals = [
        f["llm_judge_score"]
        for f in faithfulness
        if f.get("llm_judge_score") is not None
    ]
    avg_hall   = sum(hall_vals) / len(hall_vals) if hall_vals else 0.0
    avg_kw     = sum(kw_vals)   / len(kw_vals)   if kw_vals   else 1.0
    avg_judge: Optional[float] = (
        sum(judge_vals) / len(judge_vals) if judge_vals else None
    )
    faith_ok = avg_hall <= t["hallucination_proxy_max"] and avg_kw >= t["keyword_recall_min"]
    if avg_judge is not None:
        faith_ok = faith_ok and avg_judge >= t["llm_judge_score_min"]

    # ── Coverage ──────────────────────────────────────────────────────────────
    coverage_ok = (
        coverage.get("topic_coverage_rate", 0.0) >= t["topic_coverage_min"]
    )

    # ── Latency SLO ───────────────────────────────────────────────────────────
    latency_ok = lat_report.all_slos_passed()

    all_pass = chunking_ok and retrieval_ok and faith_ok and coverage_ok and latency_ok

    return {
        "pass":                    all_pass,
        "chunking_ok":             chunking_ok,
        "retrieval_ok":            retrieval_ok,
        "faithfulness_ok":         faith_ok,
        "coverage_ok":             coverage_ok,
        "latency_slo_ok":          latency_ok,
        # Helpful at-a-glance diagnostics
        "avg_mrr":                 round(avg_mrr, 4),
        "avg_hit_rate":            round(avg_hit_rate, 4),
        "avg_hallucination_proxy": round(avg_hall, 4),
        "avg_keyword_recall":      round(avg_kw, 4),
        "avg_llm_judge_score":     round(avg_judge, 4) if avg_judge is not None else None,
    }
