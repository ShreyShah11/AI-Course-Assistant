"""
Coverage Metrics
================
Evaluates whether the retrieval pipeline surfaces content from the right
source namespaces and covers the intended topics for a given query set.

Metrics
-------
  SourceNamespaceCoverage : which namespaces returned ≥1 result across the test set
  TopicCoverage           : fraction of topic tags from course syllabus represented
  ModeAdaptation          : do quiz/summarize modes actually shift source weights?
  PlannerSourceSelection  : does the planner correctly include/exclude sources?
  DifficultyDistribution  : for quiz mode — are difficulty levels requested correctly?
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


ALL_SOURCES = {"qna", "image", "audio", "youtube", "documents", "handwritten"}


@dataclass
class CoverageResult:
    source_namespace_coverage: dict[str, bool] = field(default_factory=dict)
    topic_coverage_rate: float = 0.0
    mode_adaptation_score: float = 0.0
    planner_source_precision: float = 0.0
    planner_source_recall: float = 0.0
    difficulty_distribution_error: float = 0.0
    details: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "source_namespace_coverage":   self.source_namespace_coverage,
            "topic_coverage_rate":         round(self.topic_coverage_rate, 4),
            "mode_adaptation_score":       round(self.mode_adaptation_score, 4),
            "planner_source_precision":    round(self.planner_source_precision, 4),
            "planner_source_recall":       round(self.planner_source_recall, 4),
            "difficulty_distribution_error": round(self.difficulty_distribution_error, 4),
            **self.details,
        }


class CoverageMetrics:

    @staticmethod
    def source_namespace_coverage(
        all_results: list[list[dict]],
    ) -> dict[str, bool]:
        """
        Given multiple retrieval results (one list of chunk dicts per query),
        report whether each source namespace appeared at least once across
        the entire test set.
        """
        seen: set[str] = set()
        for chunks in all_results:
            for chunk in chunks:
                sk = chunk.get("source_key", "")
                if sk:
                    seen.add(sk)
        return {src: (src in seen) for src in ALL_SOURCES}

    @staticmethod
    def topic_coverage(
        retrieved_topics: list[str],
        syllabus_topics: list[str],
    ) -> float:
        """
        Fraction of course syllabus topics that appear in retrieved chunk topics
        across the test set.
        """
        if not syllabus_topics:
            return 1.0
        syllabus_lower = {t.lower() for t in syllabus_topics}
        retrieved_lower = {t.lower() for t in retrieved_topics}
        # Partial match: a topic is "covered" if its words appear in any retrieved topic
        covered = 0
        for syl_topic in syllabus_lower:
            syl_words = set(syl_topic.split())
            if any(
                len(syl_words & set(ret.split())) / len(syl_words) >= 0.5
                for ret in retrieved_lower
            ):
                covered += 1
        return covered / len(syllabus_lower)

    @staticmethod
    def mode_adaptation_score(
        ask_source_weights: dict[str, float],
        quiz_source_weights: dict[str, float],
        summarize_source_weights: dict[str, float],
    ) -> float:
        """
        Measures how different the source priority weights are across modes.
        Expected: quiz should have high qna weight, summarize high documents+audio.

        Returns 0.0 (no adaptation) to 1.0 (fully adapted).

        Heuristic scoring:
          - quiz qna weight > ask qna weight by ≥ 0.3  → +1
          - summarize documents weight > quiz documents weight by ≥ 0.2 → +1
          - quiz youtube weight < ask youtube weight by ≥ 0.1 → +1
        """
        score = 0
        checks = 3

        quiz_qna  = quiz_source_weights.get("qna", 0.0)
        ask_qna   = ask_source_weights.get("qna", 0.0)
        if quiz_qna - ask_qna >= 0.3:
            score += 1

        sum_docs  = summarize_source_weights.get("documents", 0.0)
        quiz_docs = quiz_source_weights.get("documents", 0.0)
        if sum_docs - quiz_docs >= 0.2:
            score += 1

        quiz_yt = quiz_source_weights.get("youtube", 0.0)
        ask_yt  = ask_source_weights.get("youtube", 0.0)
        if ask_yt - quiz_yt >= 0.1:
            score += 1

        return score / checks

    @staticmethod
    def planner_source_selection(
        planned_sources: list[str],
        expected_sources: list[str],
        excluded_sources: list[str] | None = None,
    ) -> tuple[float, float]:
        """
        Precision and Recall of planner source selection.

        planned_sources  : source_key list from QueryPlan.source_budgets
        expected_sources : sources that SHOULD be in the plan
        excluded_sources : sources that should NOT be in the plan (if known)

        Returns (precision, recall).
        """
        planned_set  = set(planned_sources)
        expected_set = set(expected_sources)
        excluded_set = set(excluded_sources or [])

        if not planned_set:
            return 0.0, 0.0

        precision = len(planned_set & expected_set) / len(planned_set) if planned_set else 0.0
        recall    = len(planned_set & expected_set) / len(expected_set) if expected_set else 1.0

        # Penalise including explicitly excluded sources
        penalty = len(planned_set & excluded_set) / len(planned_set) if planned_set else 0.0
        precision = max(0.0, precision - penalty)

        return round(precision, 4), round(recall, 4)

    @staticmethod
    def difficulty_distribution_error(
        quiz_chunks: list[dict],
        target_distribution: dict[str, float] | None = None,
    ) -> float:
        """
        For quiz mode: measures L1 distance between the difficulty distribution
        of retrieved QnA chunks and the target distribution from QuizConfig.

        quiz_chunks        : retrieved chunk dicts (source_key=qna)
        target_distribution: {"easy": 0.3, "medium": 0.5, "hard": 0.2}

        Returns L1 error (lower is better; 0.0 = perfect match).
        """
        if target_distribution is None:
            target_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}

        qna_chunks = [c for c in quiz_chunks if c.get("source_key") == "qna"]
        if not qna_chunks:
            return 1.0  # no QnA chunks retrieved for a quiz query = bad

        diff_counts: dict[str, int] = Counter(
            c.get("difficulty", "unknown") for c in qna_chunks
        )
        total = len(qna_chunks)
        actual: dict[str, float] = {k: v / total for k, v in diff_counts.items()}

        l1 = sum(
            abs(target_distribution.get(d, 0.0) - actual.get(d, 0.0))
            for d in set(list(target_distribution.keys()) + list(actual.keys()))
        )
        return min(l1, 1.0)

    @classmethod
    def evaluate_suite(
        cls,
        suite_results: list[dict],
        syllabus_topics: list[str] | None = None,
    ) -> CoverageResult:
        """
        Evaluate coverage across a full test suite.

        Parameters
        ----------
        suite_results : list of dicts, each with:
            {
              "mode": str,
              "chunks": list[dict],              # retrieved chunk .to_dict()
              "plan": {                           # QueryPlan fields
                "source_budgets": [...],
                "quiz_config": {...} | None,
              },
              "expected_sources": list[str],
              "excluded_sources": list[str],
            }
        syllabus_topics : list of topic strings from course syllabus
        """
        all_chunk_lists = [r["chunks"] for r in suite_results]
        ns_coverage = cls.source_namespace_coverage(all_chunk_lists)

        # Topic coverage
        all_topics = [
            c.get("topic", "")
            for r in suite_results
            for c in r["chunks"]
        ]
        topic_cov = cls.topic_coverage(all_topics, syllabus_topics or [])

        # Planner source selection (aggregate precision/recall)
        prec_list, rec_list = [], []
        for r in suite_results:
            budgets = r.get("plan", {}).get("source_budgets", [])
            planned = [b.get("source_key", "") if isinstance(b, dict) else b.source_key
                       for b in budgets]
            expected = r.get("expected_sources", [])
            excluded = r.get("excluded_sources", [])
            p, rec = cls.planner_source_selection(planned, expected, excluded)
            prec_list.append(p)
            rec_list.append(rec)

        # Difficulty distribution (quiz mode only)
        quiz_results = [r for r in suite_results if r.get("mode") == "quiz"]
        diff_errors = []
        for qr in quiz_results:
            target_dist = (
                qr.get("plan", {}).get("quiz_config", {}) or {}
            ).get("difficulty_distribution")
            err = cls.difficulty_distribution_error(qr["chunks"], target_dist)
            diff_errors.append(err)

        return CoverageResult(
            source_namespace_coverage=ns_coverage,
            topic_coverage_rate=topic_cov,
            planner_source_precision=statistics.mean(prec_list) if prec_list else 0.0,
            planner_source_recall=statistics.mean(rec_list) if rec_list else 0.0,
            difficulty_distribution_error=statistics.mean(diff_errors) if diff_errors else 0.0,
            details={
                "num_queries": len(suite_results),
                "num_quiz_queries": len(quiz_results),
                "namespaces_seen": [k for k, v in ns_coverage.items() if v],
            },
        )