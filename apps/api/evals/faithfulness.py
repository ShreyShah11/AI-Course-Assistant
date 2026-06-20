"""
Faithfulness & Answer Quality Metrics
======================================
These metrics evaluate how well the final LLM answer is grounded in
the retrieved chunks. They run on the answer text + context string
WITHOUT requiring ground-truth answers (reference-free evaluation).

Metrics
-------
  ContextUtilisation  : fraction of retrieved chunks referenced in answer
  CitationCoverage    : fraction of [N] citation markers present in answer
  AnswerLengthRatio   : answer length vs expected_answer_length hint
  KeywordRecall       : domain_keywords from QueryPlan present in answer
  SourceAttributionBalance : distribution of [N] markers (is one source dominating?)
  HallucinationProxy  : sentences in answer that share NO token overlap with context
                        (rough proxy — higher = risk of hallucination)
  PlanAlignmentScore  : how well answer format matches expected_answer_format
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field


@dataclass
class FaithfulnessResult:
    context_utilisation: float = 0.0       # 0-1: fraction of chunks cited
    citation_coverage: float = 0.0         # 0-1: completeness of [N] markers
    answer_length_ok: bool = True
    keyword_recall: float = 0.0            # 0-1
    source_attribution_balance: float = 0.0 # Shannon entropy (0-1)
    hallucination_proxy: float = 0.0       # lower is better (0-1)
    plan_alignment_score: float = 0.0      # 0-1
    details: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "context_utilisation":        round(self.context_utilisation, 4),
            "citation_coverage":          round(self.citation_coverage, 4),
            "answer_length_ok":           self.answer_length_ok,
            "keyword_recall":             round(self.keyword_recall, 4),
            "source_attribution_balance": round(self.source_attribution_balance, 4),
            "hallucination_proxy":        round(self.hallucination_proxy, 4),
            "plan_alignment_score":       round(self.plan_alignment_score, 4),
        }


class FaithfulnessMetrics:

    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")
    _STOPWORDS = {
        "a","an","the","is","in","on","of","to","for","and","or","with",
        "that","this","it","as","at","be","by","from","its","was","are",
        "were","has","have","had","not","but","can","will","just","do",
    }

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        return {t for t in tokens if t not in cls._STOPWORDS}

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if len(s.split()) >= 4]

    @classmethod
    def context_utilisation(cls, answer: str, num_chunks: int) -> float:
        """Fraction of retrieved chunks cited at least once in the answer."""
        if num_chunks == 0:
            return 0.0
        cited_indices = {int(m) for m in cls._CITATION_PATTERN.findall(answer)}
        # Filter to valid indices (1-based up to num_chunks)
        valid_citations = {i for i in cited_indices if 1 <= i <= num_chunks}
        return len(valid_citations) / num_chunks

    @classmethod
    def citation_coverage(cls, answer: str, num_chunks: int) -> float:
        """
        Are citations dense enough? Counts sentences that have at least
        one [N] citation vs total non-trivial sentences.
        """
        sentences = cls._split_sentences(answer)
        if not sentences:
            return 0.0
        cited_sentences = sum(1 for s in sentences if cls._CITATION_PATTERN.search(s))
        return cited_sentences / len(sentences)

    @classmethod
    def answer_length_ok(
        cls, answer: str, expected_length: str
    ) -> tuple[bool, int]:
        """
        expected_length: "short" | "medium" | "long"
        Returns (within_range, actual_word_count)
        """
        word_count = len(answer.split())
        ranges = {
            "short":  (40,  350),
            "medium": (200, 900),
            "long":   (500, 3000),
        }
        lo, hi = ranges.get(expected_length, (0, 99_999))
        return lo <= word_count <= hi, word_count

    @classmethod
    def keyword_recall(cls, answer: str, domain_keywords: list[str]) -> float:
        """Fraction of QueryPlan domain_keywords that appear in the answer."""
        if not domain_keywords:
            return 1.0
        answer_lower = answer.lower()
        found = sum(1 for kw in domain_keywords if kw.lower() in answer_lower)
        return found / len(domain_keywords)

    @classmethod
    def source_attribution_balance(cls, answer: str) -> float:
        """
        Shannon entropy of citation distribution.
        High → citations spread across sources (good).
        Near 0 → all citations point to one source.
        """
        import math
        matches = cls._CITATION_PATTERN.findall(answer)
        if not matches:
            return 0.0
        from collections import Counter
        counts = Counter(matches)
        total = sum(counts.values())
        entropy = -sum((v / total) * math.log2(v / total) for v in counts.values())
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @classmethod
    def hallucination_proxy(cls, answer: str, context_string: str) -> float:
        """
        Rough proxy: fraction of answer sentences with NO token overlap
        with the retrieved context. Higher rate = more suspicious.

        Note: This is NOT a true hallucination detector; it's a signal.
        Sentences introducing topics or transitions may correctly have low overlap.
        Use alongside LLM-based evaluation for production.
        """
        sentences = cls._split_sentences(answer)
        if not sentences:
            return 0.0
        context_tokens = cls._tokenize(context_string)
        no_overlap_count = 0
        for sent in sentences:
            sent_tokens = cls._tokenize(sent)
            if not sent_tokens & context_tokens:
                no_overlap_count += 1
        return no_overlap_count / len(sentences)

    _FORMAT_SIGNALS = {
        "structured_explanation": [r"##\s+", r"\*\*", r"\n\n"],
        "definition_then_example": [r"(?:defined as|definition|means that)", r"(?:for example|e\.g\.|instance)"],
        "comparison_table": [r"\|.*\|", r"vs\.?\s", r"compared to"],
        "worked_solution": [r"(?:step \d|given:|find:|solution:)", r"=\s*\d"],
        "quiz_questions": [r"^\d+[.)]\s", r"(?:answer|correct answer|explanation):"],
        "topic_summary": [r"##\s+", r"overview|summary|key points"],
        "bullet_summary": [r"^[\-\*]\s", r"•\s"],
        "conceptual_prose": [],  # no specific markers needed
    }

    @classmethod
    def plan_alignment_score(cls, answer: str, expected_format: str) -> float:
        """
        Checks whether the answer contains structural markers consistent
        with the expected_answer_format from the QueryPlan.
        """
        patterns = cls._FORMAT_SIGNALS.get(expected_format, [])
        if not patterns:
            return 1.0  # conceptual_prose has no required markers
        hits = sum(
            1 for p in patterns
            if re.search(p, answer, re.MULTILINE | re.IGNORECASE)
        )
        return hits / len(patterns)

    @classmethod
    def evaluate(
        cls,
        answer: str,
        context_string: str,
        num_chunks: int,
        domain_keywords: list[str],
        expected_answer_length: str = "medium",
        expected_answer_format: str = "structured_explanation",
    ) -> FaithfulnessResult:
        """
        Run all faithfulness metrics and return a FaithfulnessResult.

        Parameters
        ----------
        answer                 : the LLM-generated answer text
        context_string         : full context string passed to the LLM
                                 (RetrievalResult.to_context_string())
        num_chunks             : number of retrieved chunks (for citation checks)
        domain_keywords        : from QueryPlan.domain_keywords
        expected_answer_length : from QueryPlan.expected_answer_length
        expected_answer_format : from QueryPlan.expected_answer_format
        """
        length_ok, word_count = cls.answer_length_ok(answer, expected_answer_length)

        result = FaithfulnessResult(
            context_utilisation=cls.context_utilisation(answer, num_chunks),
            citation_coverage=cls.citation_coverage(answer, num_chunks),
            answer_length_ok=length_ok,
            keyword_recall=cls.keyword_recall(answer, domain_keywords),
            source_attribution_balance=cls.source_attribution_balance(answer),
            hallucination_proxy=cls.hallucination_proxy(answer, context_string),
            plan_alignment_score=cls.plan_alignment_score(answer, expected_answer_format),
            details={
                "word_count": word_count,
                "expected_length": expected_answer_length,
                "citation_indices": sorted(
                    {int(m) for m in cls._CITATION_PATTERN.findall(answer)}
                ),
            },
        )
        return result