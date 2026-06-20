"""
runner_live.py
==============
LIVE end-to-end RAG pipeline evaluator driven by Test suites/tests.py.

Flow
────
For each RETRIEVAL_SCENARIO in tests.py:
  1.  Call pipeline.retrieve() with the scenario's real query + course_id
      → real Pinecone dense search + BM25 + MMR + RRF
  2.  Generate the final Gemini answer (same code as /retrieval/ask)
  3.  Measure wall-clock latency per stage
  4.  Evaluate the REAL returned chunks against ground-truth from the test suite:
        ▸ Chunking   – size OK, metadata completeness, text length
        ▸ Retrieval  – MRR and NDCG@K using scenario.relevant_chunk_ids
                       Hit-rate@K, Precision@K, source diversity
        ▸ Faithfulness – hallucination proxy, keyword recall, citation density
        ▸ Coverage   – expected namespaces actually present in results
  5.  Optionally run Gemini LLM-as-judge on each real (query, context, answer)
  6.  Return a structured LiveEvalReport as JSON

The chunk fixtures (QNA_CHUNKS, AUDIO_CHUNKS, …) in tests.py are used to
validate the *shape* of returned chunks (metadata schema checks).
The RETRIEVAL_SCENARIOS are the authoritative ground-truth source for
relevance judgements against real Pinecone results.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Load the real retrieval pipeline (handles space in directory name)
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE = None


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    api_root = Path(__file__).resolve().parents[1]
    pipeline_file = api_root / "pipelines" / "retrieval pipeline" / "agentic_retrieval.py"
    if not pipeline_file.exists():
        raise FileNotFoundError(f"Pipeline not found: {pipeline_file}")
    project_root = api_root.parents[1]
    for p in (str(project_root), str(api_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("_live_pipeline", pipeline_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _PIPELINE = mod
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Load test suite fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _load_test_suites():
    """Load RETRIEVAL_SCENARIOS and chunk fixtures from tests.py."""
    suite_file = Path(__file__).resolve().parent / "Test suites" / "tests.py"
    spec = importlib.util.spec_from_file_location("_test_suites", suite_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Expected metadata fields per source type (for schema validation)
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: dict[str, list[str]] = {
    "qna":         ["chunk_type", "topic", "source_file"],
    "image":       ["source_file", "course_id"],
    "audio":       ["strategy", "lecture_title"],
    "youtube":     ["video_title", "deep_link"],
    "documents":   ["source_file", "section_title"],
    "handwritten": ["source_file", "topic"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tok(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]{3,}", text.lower()))


# ── Chunking quality ──────────────────────────────────────────────────────────

def _meta_completeness(chunk) -> float:
    sk = getattr(chunk, "source_key", "documents")
    required = _REQUIRED_FIELDS.get(sk, ["source_file"])
    present = sum(1 for f in required if getattr(chunk, f, None))
    return round(present / len(required), 4) if required else 1.0


def _size_ok(chunk) -> bool:
    words = len((getattr(chunk, "text", "") or "").split())
    return 15 <= words <= 2000


def _text_richness(chunk) -> float:
    """Fraction of unique words (type-token ratio proxy, capped at 1)."""
    words = (getattr(chunk, "text", "") or "").lower().split()
    if not words:
        return 0.0
    return round(min(1.0, len(set(words)) / len(words) + 0.1), 4)


def _chunking_report(chunks: list) -> dict:
    if not chunks:
        return {"total": 0, "avg_meta_completeness": 0.0, "pct_size_ok": 0.0,
                "avg_richness": 0.0, "source_breakdown": {}}
    mc = [_meta_completeness(c) for c in chunks]
    sz = [_size_ok(c) for c in chunks]
    ri = [_text_richness(c) for c in chunks]
    bd: dict[str, int] = {}
    for c in chunks:
        sk = getattr(c, "source_key", "unknown")
        bd[sk] = bd.get(sk, 0) + 1
    return {
        "total": len(chunks),
        "avg_meta_completeness": round(sum(mc) / len(mc), 4),
        "pct_size_ok": round(sum(sz) / len(sz), 4),
        "avg_richness": round(sum(ri) / len(ri), 4),
        "source_breakdown": bd,
    }


# ── Retrieval quality against ground truth ───────────────────────────────────

def _mrr(chunk_ids: list[str], relevant: set[str]) -> float:
    for rank, cid in enumerate(chunk_ids, 1):
        if cid in relevant:
            return round(1.0 / rank, 4)
    return 0.0


def _dcg(chunk_ids: list[str], grades: dict[str, int], k: int) -> float:
    score = 0.0
    for rank, cid in enumerate(chunk_ids[:k], 1):
        rel = grades.get(cid, 0)
        score += rel / math.log2(rank + 1)
    return score


def _ndcg(chunk_ids: list[str], grades: dict[str, int], k: int) -> float:
    actual_dcg = _dcg(chunk_ids, grades, k)
    ideal_ids = sorted(grades, key=grades.get, reverse=True)
    ideal_dcg = _dcg(ideal_ids, grades, k)
    return round(actual_dcg / ideal_dcg, 4) if ideal_dcg > 0 else 0.0


def _precision_at_k(chunk_ids: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for cid in chunk_ids[:k] if cid in relevant)
    return round(hits / min(k, len(chunk_ids)), 4) if chunk_ids else 0.0


def _hit_rate(chunk_ids: list[str], relevant: set[str], k: int = 5) -> float:
    return 1.0 if any(cid in relevant for cid in chunk_ids[:k]) else 0.0


def _source_coverage(chunks: list, expected_sources: set[str]) -> float:
    """Fraction of expected source namespaces actually present in results."""
    if not expected_sources:
        return 1.0
    found = {getattr(c, "source_key", "") for c in chunks}
    return round(len(found & expected_sources) / len(expected_sources), 4)


def _source_diversity(chunks: list) -> float:
    """Fraction of all 6 namespaces present."""
    namespaces = {getattr(c, "source_key", "") for c in chunks}
    return round(len(namespaces) / 6.0, 4)


# ── Faithfulness ──────────────────────────────────────────────────────────────

def _hallucination_proxy(answer: str, context: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 15]
    if not sentences:
        return 0.0
    ctx_tok = _tok(context)
    grounded = sum(1 for s in sentences if _tok(s) & ctx_tok)
    return round(1.0 - grounded / len(sentences), 4)


def _keyword_recall(answer: str, domain_kw: list[str]) -> float:
    if not domain_kw:
        return 1.0
    ans_lower = answer.lower()
    return round(sum(1 for kw in domain_kw if kw.lower() in ans_lower) / len(domain_kw), 4)


def _citation_density(answer: str, n_chunks: int) -> float:
    if not n_chunks:
        return 0.0
    cited = set(re.findall(r"\[(\d+)\]", answer))
    return round(len(cited) / n_chunks, 4)


# ── LLM judge ────────────────────────────────────────────────────────────────

def _llm_judge(query: str, context: str, answer: str, model: str) -> Optional[float]:
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or not model:
            return None
        client = genai.Client(api_key=api_key.strip())
        prompt = (
            "Rate how faithfully the ANSWER is grounded in the CONTEXT.\n\n"
            f"QUERY: {query}\n\n"
            f"CONTEXT (first 3000 chars):\n{context[:3000]}\n\n"
            f"ANSWER (first 1500 chars):\n{answer[:1500]}\n\n"
            "Return ONLY valid JSON: {\"score\": <0.0-1.0>, \"reason\": \"<one sentence>\"}\n"
            "1.0 = every claim is fully grounded. 0.0 = completely hallucinated."
        )
        resp = client.models.generate_content(model=model, contents=prompt)
        m = re.search(r"\{.*?\}", (resp.text or ""), re.S)
        if not m:
            return None
        data = json.loads(m.group())
        return round(max(0.0, min(1.0, float(data.get("score", 0.5)))), 4)
    except Exception as e:
        logger.warning("LLM judge failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveChunkingMetrics:
    total_chunks: int
    avg_meta_completeness: float
    pct_size_ok: float
    avg_richness: float
    source_breakdown: dict


@dataclass
class LiveRetrievalMetrics:
    chunks_returned: int
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    precision_at_5: float
    hit_rate_at_5: float
    source_coverage: float       # fraction of expected_sources found
    source_diversity: float      # fraction of all 6 namespaces
    namespaces_hit: list[str]
    relevant_found: list[str]    # ground-truth IDs that actually appeared


@dataclass
class LiveFaithfulnessMetrics:
    hallucination_proxy: float
    keyword_recall: float
    citation_density: float
    llm_judge_score: Optional[float]


@dataclass
class LiveQueryResult:
    scenario_id: str
    query: str
    mode: str
    course_id: str
    latency_ms: float
    chunking: LiveChunkingMetrics
    retrieval: LiveRetrievalMetrics
    faithfulness: LiveFaithfulnessMetrics
    answer_preview: str
    error: Optional[str] = None


@dataclass
class LiveOverallHealth:
    pass_eval: bool
    scenarios_run: int
    scenarios_errored: int
    # Latency
    avg_latency_ms: float
    latency_slo_ok: bool         # avg < 20 000 ms
    # Retrieval
    avg_mrr: float
    avg_ndcg_at_5: float
    avg_hit_rate_at_5: float
    avg_source_coverage: float
    namespaces_seen: list[str]
    retrieval_ok: bool           # avg_mrr >= 0.2 AND avg_chunks > 0
    # Chunking
    avg_meta_completeness: float
    chunking_ok: bool            # >= 0.7
    # Faithfulness
    avg_hallucination_proxy: float
    avg_keyword_recall: float
    avg_citation_density: float
    avg_llm_judge_score: Optional[float]
    faithfulness_ok: bool        # avg_hallucination_proxy < 0.45


@dataclass
class LiveEvalReport:
    course_id: str
    top_k: int
    seed_report: Optional[dict]
    scenarios: list[LiveQueryResult]
    overall_health: LiveOverallHealth
    ran_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_live_eval(
    course_id: str,
    queries: Optional[list[dict]] = None,   # ignored — always uses test suite
    include_llm_judge: bool = False,
    top_k: int = 8,
    seed_first: bool = True,
) -> dict:
    """
    Seed the Pinecone index with test-suite fixture chunks, then evaluate.

    Flow
    ----
    1.  (seed_first=True) Embeds all fixture chunks from tests.py via Gemini
        and upserts them into the `course_id` Pinecone index under their
        correct namespaces (qna-chunks, audio-chunks, document-chunks, etc.).
        The index is created automatically if it doesn't exist.
    2.  Waits 3 s for Pinecone to make the vectors queryable.
    3.  Runs each RETRIEVAL_SCENARIO through pipeline.retrieve() — real
        Pinecone ANN + BM25 + MMR + RRF against the just-seeded data.
    4.  Generates a real Gemini answer for each scenario.
    5.  Evaluates all results against ground truth from tests.py.

    Parameters
    ----------
    course_id : str
        Pinecone index name to seed and query (e.g. "Test01").
        Created automatically if it doesn't exist.
    include_llm_judge : bool
        Call EVAL_LLM_JUDGE_MODEL for each scenario if True.
    top_k : int
        Chunks to retrieve per query.
    seed_first : bool
        Default True. Set False to skip seeding (reuse a previously seeded index).
    """
    from routes.agentic_retrieval import generate_final_response

    pipeline = _load_pipeline()
    suites = _load_test_suites()
    RetrievalMode = pipeline.RetrievalMode
    judge_model = os.getenv("EVAL_LLM_JUDGE_MODEL", "") if include_llm_judge else ""

    # ── Step 0: Seed the Pinecone index with fixture chunks ───────────────────
    seed_report: Optional[dict] = None
    if seed_first:
        logger.info("Seeding Pinecone index '%s' with test-suite fixtures…", course_id)
        from evals.seed_pinecone import seed_eval_index
        seed_report = seed_eval_index(course_id=course_id)
        logger.info(
            "Seeding complete: %d vectors across %d sources",
            seed_report["total_upserted"],
            len(seed_report["per_source"]),
        )
        time.sleep(3)  # let Pinecone make vectors queryable

    scenarios: list[dict] = suites.RETRIEVAL_SCENARIOS
    results: list[LiveQueryResult] = []

    for sc in scenarios:
        q = sc["query"]
        mode = sc.get("mode", "ask")
        relevant_ids: set[str] = sc.get("relevant_chunk_ids", set())
        expected_sources: set[str] = sc.get("expected_sources", set())
        grades: dict[str, int] = sc.get("relevance_grades", {})

        t0 = time.perf_counter()
        error: Optional[str] = None
        result = None

        # ── Step 1: Real retrieval ────────────────────────────────────────────
        try:
            result = pipeline.retrieve(
                query=q,
                course_id=course_id,
                mode=RetrievalMode(mode),
                top_k=top_k,
                verbose=False,
            )
        except Exception as exc:
            error = f"retrieval failed: {exc}"

        # ── Step 2: Real Gemini answer ────────────────────────────────────────
        final_answer = ""
        if result and result.chunks and not error:
            try:
                answer_prompt = result.to_answer_prompt()
                final_answer, _ = generate_final_response(
                    answer_prompt=answer_prompt, mode=mode
                )
            except Exception as exc:
                error = f"answer generation failed: {exc}"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        if error and result is None:
            results.append(LiveQueryResult(
                scenario_id=sc.get("id", q[:40]),
                query=q, mode=mode, course_id=course_id,
                latency_ms=latency_ms,
                chunking=LiveChunkingMetrics(0, 0.0, 0.0, 0.0, {}),
                retrieval=LiveRetrievalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], []),
                faithfulness=LiveFaithfulnessMetrics(1.0, 0.0, 0.0, None),
                answer_preview="", error=error,
            ))
            continue

        chunks = result.chunks if result else []
        chunk_ids = [c.chunk_id for c in chunks]

        # ── Chunking quality on real chunks ───────────────────────────────────
        cr = _chunking_report(chunks)
        chunking = LiveChunkingMetrics(
            total_chunks=cr["total"],
            avg_meta_completeness=cr["avg_meta_completeness"],
            pct_size_ok=cr["pct_size_ok"],
            avg_richness=cr["avg_richness"],
            source_breakdown=cr["source_breakdown"],
        )

        # ── Retrieval metrics vs ground truth from tests.py ───────────────────
        mrr_score       = _mrr(chunk_ids, relevant_ids)
        ndcg5           = _ndcg(chunk_ids, grades, k=5)
        ndcg10          = _ndcg(chunk_ids, grades, k=10)
        p_at_5          = _precision_at_5 = _precision_at_k(chunk_ids, relevant_ids, k=5)
        hit_rate_5      = _hit_rate(chunk_ids, relevant_ids, k=5)
        src_coverage    = _source_coverage(chunks, expected_sources)
        src_diversity   = _source_diversity(chunks)
        namespaces_hit  = sorted({getattr(c, "source_key", "") for c in chunks})
        relevant_found  = [cid for cid in chunk_ids if cid in relevant_ids]

        retrieval = LiveRetrievalMetrics(
            chunks_returned=len(chunks),
            mrr=mrr_score,
            ndcg_at_5=ndcg5,
            ndcg_at_10=ndcg10,
            precision_at_5=p_at_5,
            hit_rate_at_5=hit_rate_5,
            source_coverage=src_coverage,
            source_diversity=src_diversity,
            namespaces_hit=namespaces_hit,
            relevant_found=relevant_found,
        )

        # ── Faithfulness of real answer vs real retrieved context ─────────────
        context_str = result.to_context_string() if chunks else ""
        domain_kw = list(sc.get("planner_data", {}).get("domain_keywords", []))
        if result and result.plan:
            domain_kw = domain_kw or list(result.plan.domain_keywords or [])

        hallucination   = _hallucination_proxy(final_answer, context_str)
        kw_recall       = _keyword_recall(final_answer, domain_kw)
        citation_dens   = _citation_density(final_answer, len(chunks))
        judge_score: Optional[float] = None
        if judge_model and final_answer and context_str:
            judge_score = _llm_judge(q, context_str, final_answer, judge_model)

        faithfulness = LiveFaithfulnessMetrics(
            hallucination_proxy=hallucination,
            keyword_recall=kw_recall,
            citation_density=citation_dens,
            llm_judge_score=judge_score,
        )

        results.append(LiveQueryResult(
            scenario_id=sc.get("id", q[:40]),
            query=q, mode=mode, course_id=course_id,
            latency_ms=latency_ms,
            chunking=chunking,
            retrieval=retrieval,
            faithfulness=faithfulness,
            answer_preview=final_answer[:400],
            error=error,
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # Overall health
    # ─────────────────────────────────────────────────────────────────────────
    ok = [r for r in results if r.error is None]
    n_ok = len(ok)
    n_err = len(results) - n_ok

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    avg_latency     = avg([r.latency_ms for r in ok])
    avg_mrr         = avg([r.retrieval.mrr for r in ok])
    avg_ndcg5       = avg([r.retrieval.ndcg_at_5 for r in ok])
    avg_hit5        = avg([r.retrieval.hit_rate_at_5 for r in ok])
    avg_src_cov     = avg([r.retrieval.source_coverage for r in ok])
    avg_meta        = avg([r.chunking.avg_meta_completeness for r in ok])
    avg_halluc      = avg([r.faithfulness.hallucination_proxy for r in ok])
    avg_kw_rec      = avg([r.faithfulness.keyword_recall for r in ok])
    avg_citation    = avg([r.faithfulness.citation_density for r in ok])
    avg_chunks      = avg([r.retrieval.chunks_returned for r in ok])
    judge_scores    = [r.faithfulness.llm_judge_score for r in ok
                       if r.faithfulness.llm_judge_score is not None]
    avg_judge       = round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None
    all_ns          = sorted({ns for r in ok for ns in r.retrieval.namespaces_hit})

    latency_slo_ok  = avg_latency < 20_000
    retrieval_ok    = avg_chunks > 0 and avg_mrr >= 0.2
    chunking_ok     = avg_meta >= 0.70
    faithfulness_ok = avg_halluc < 0.45

    overall = LiveOverallHealth(
        pass_eval=latency_slo_ok and retrieval_ok and chunking_ok and faithfulness_ok and n_err == 0,
        scenarios_run=len(results),
        scenarios_errored=n_err,
        avg_latency_ms=avg_latency,
        latency_slo_ok=latency_slo_ok,
        avg_mrr=avg_mrr,
        avg_ndcg_at_5=avg_ndcg5,
        avg_hit_rate_at_5=avg_hit5,
        avg_source_coverage=avg_src_cov,
        namespaces_seen=all_ns,
        retrieval_ok=retrieval_ok,
        avg_meta_completeness=avg_meta,
        chunking_ok=chunking_ok,
        avg_hallucination_proxy=avg_halluc,
        avg_keyword_recall=avg_kw_rec,
        avg_citation_density=avg_citation,
        avg_llm_judge_score=avg_judge,
        faithfulness_ok=faithfulness_ok,
    )

    report = LiveEvalReport(
        course_id=course_id,
        top_k=top_k,
        seed_report=seed_report,
        scenarios=results,
        overall_health=overall,
        ran_at=datetime.datetime.utcnow().isoformat() + "Z",
    )

    def _ser(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _ser(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_ser(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        return obj

    return _ser(report)
