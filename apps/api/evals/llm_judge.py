"""
LLM-as-Judge for faithfulness evaluation.
==========================================
Uses Gemini (model from EVAL_LLM_JUDGE_MODEL env var) to score how
faithfully an LLM answer is grounded in its retrieved context.

If EVAL_LLM_JUDGE_MODEL is not set, or GEMINI_API_KEY is missing,
the judge is silently skipped and returns None — the eval pipeline
then falls back to the heuristic hallucination_proxy from faithfulness.py.

Env vars used
-------------
  EVAL_LLM_JUDGE_MODEL  - e.g. "gemini-2.5-flash" (required to activate judge)
  GEMINI_API_KEY        - already set in .env (reused, no new key needed)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system used in an AI course assistant.

You will be given:
1. A QUESTION asked by a student
2. RETRIEVED CONTEXT: the text chunks surfaced by the retrieval pipeline
3. AN ANSWER: what the LLM generated from those chunks

Evaluate the answer on exactly these three criteria:

A) FACTUAL_GROUNDING (0-10): Every factual claim in the answer is supported by the retrieved context.
B) NO_HALLUCINATION (0-10): The answer avoids introducing facts that are NOT present in the context.
C) CITATION_CORRECTNESS (0-10): The [N] citation markers are used correctly and consistently.

Return ONLY a valid JSON object — no markdown, no extra text — in this exact format:
{{
  "factual_grounding": <int 0-10>,
  "no_hallucination": <int 0-10>,
  "citation_correctness": <int 0-10>,
  "overall_score": <float 0.0-1.0, computed as the average of the three scores divided by 10>,
  "reasoning": "<2-3 sentence explanation>"
}}

---
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}
"""


@dataclass
class LLMJudgeResult:
    """Result from the LLM-as-judge evaluation."""
    score: float                   # 0.0 – 1.0  (normalised overall)
    factual_grounding: float       # 0.0 – 1.0
    no_hallucination: float        # 0.0 – 1.0
    citation_correctness: float    # 0.0 – 1.0
    reasoning: str
    model_used: str

    def summary(self) -> dict:
        return {
            "score":                round(self.score, 4),
            "factual_grounding":    round(self.factual_grounding, 4),
            "no_hallucination":     round(self.no_hallucination, 4),
            "citation_correctness": round(self.citation_correctness, 4),
            "reasoning":            self.reasoning,
            "model_used":           self.model_used,
        }


class LLMJudge:
    """
    Optional Gemini-based LLM judge for faithfulness evaluation.

    Usage
    -----
    if LLMJudge.is_available():
        result = LLMJudge.score(question, context, answer)
    """

    @staticmethod
    def is_available() -> bool:
        """True if both EVAL_LLM_JUDGE_MODEL and GEMINI_API_KEY are set."""
        return bool(
            os.getenv("EVAL_LLM_JUDGE_MODEL", "").strip()
            and os.getenv("GEMINI_API_KEY", "").strip()
        )

    @classmethod
    def score(
        cls,
        question: str,
        context: str,
        answer: str,
        context_max_chars: int = 6_000,
    ) -> Optional[LLMJudgeResult]:
        """
        Call Gemini to score the answer against the context.

        Returns None (gracefully) if:
          - EVAL_LLM_JUDGE_MODEL or GEMINI_API_KEY is not set
          - The API call fails for any reason

        Parameters
        ----------
        question         : the original user query
        context          : concatenated retrieved chunk text
        answer           : the LLM-generated answer to evaluate
        context_max_chars: safety cap to avoid token overflow (default 6 000)
        """
        if not cls.is_available():
            return None

        model_name = os.getenv("EVAL_LLM_JUDGE_MODEL", "gemini-2.5-flash").strip()
        api_key    = os.getenv("GEMINI_API_KEY", "").strip()

        try:
            import google.generativeai as genai  # type: ignore[import]

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)

            prompt = _JUDGE_PROMPT.format(
                question=question,
                context=context[:context_max_chars],
                answer=answer,
            )

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )

            raw = response.text.strip()

            # Strip accidental markdown fences
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:])
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]

            data = json.loads(raw)

            fg  = float(data.get("factual_grounding", 0)) / 10
            nh  = float(data.get("no_hallucination", 0)) / 10
            cc  = float(data.get("citation_correctness", 0)) / 10
            # Use model-reported overall_score if present, else recompute
            overall = float(data.get("overall_score", (fg + nh + cc) / 3))
            overall = max(0.0, min(1.0, overall))  # clamp to [0, 1]

            return LLMJudgeResult(
                score=overall,
                factual_grounding=fg,
                no_hallucination=nh,
                citation_correctness=cc,
                reasoning=data.get("reasoning", ""),
                model_used=model_name,
            )

        except Exception as exc:
            logger.warning(
                "LLM judge call failed — skipping and returning None. Reason: %s", exc
            )
            return None
