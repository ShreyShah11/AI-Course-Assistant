"""
Test Fixtures
=============
Synthetic, deterministic fixtures for all six pipeline types.
No live API calls needed — these fixtures mirror the exact metadata schemas
produced by each chunking pipeline.

Usage
-----
from evals.fixtures.fixtures import (
    QNA_CHUNKS, AUDIO_CHUNKS, DOCUMENT_CHUNKS,
    IMAGE_CHUNKS, HANDWRITTEN_CHUNKS, YOUTUBE_CHUNKS,
    RETRIEVAL_SCENARIOS,
)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# QnA Pipeline Fixtures  (QnAChunk.asdict() format)
# ─────────────────────────────────────────────────────────────────────────────

QNA_CHUNKS: list[dict] = [
    {
        "chunk_id": "qna_001",
        "source_file": "CS301_PYQ_2022.pdf",
        "file_type": "pdf",
        "page_range": [1, 2],
        "chunk_type": "single_qa",
        "topic_cluster": 0,
        "topic_label": "dynamic programming / optimal substructure",
        "questions": [
            "Explain the principle of optimal substructure with an example. [8 marks]"
        ],
        "answers": [
            "A problem exhibits optimal substructure if an optimal solution to the problem "
            "contains within it optimal solutions to subproblems. Example: shortest path "
            "in a graph — if p = p1 + v + p2 is shortest from u to w, then p1 is shortest "
            "from u to v and p2 is shortest from v to w."
        ],
        "raw_text": (
            "Q1. Explain the principle of optimal substructure with an example. [8 marks]\n"
            "Answer: A problem exhibits optimal substructure..."
        ),
        "char_count": 412,
        "token_estimate": 103,
        "difficulty_hint": "medium",
        "has_sub_parts": False,
        "marks_hint": 8,
        "year_hint": 2022,
    },
    {
        "chunk_id": "qna_002",
        "source_file": "CS301_PYQ_2022.pdf",
        "file_type": "pdf",
        "page_range": [2, 3],
        "chunk_type": "multi_qa",
        "topic_cluster": 0,
        "topic_label": "dynamic programming / memoization",
        "questions": [
            "a) Define memoization and tabulation.",
            "b) Compare their time and space complexities.",
        ],
        "answers": [
            "a) Memoization stores results of expensive function calls (top-down DP).",
            "b) Both are O(n) time; memoization uses call-stack space, tabulation is iterative.",
        ],
        "raw_text": (
            "Q2. a) Define memoization and tabulation. b) Compare complexities."
        ),
        "char_count": 280,
        "token_estimate": 70,
        "difficulty_hint": "medium",
        "has_sub_parts": True,
        "marks_hint": 6,
        "year_hint": 2022,
    },
    {
        "chunk_id": "qna_003",
        "source_file": "CS301_PYQ_2021.pdf",
        "file_type": "pdf",
        "page_range": [5, 5],
        "chunk_type": "single_qa",
        "topic_cluster": 1,
        "topic_label": "graph algorithms / Dijkstra",
        "questions": [
            "Apply Dijkstra's algorithm to the graph below and find the shortest path "
            "from vertex A to vertex F. [10 marks]"
        ],
        "answers": [
            "Step 1: Initialize distances. A=0, all others=∞. Step 2: Visit A, relax edges..."
        ],
        "raw_text": "Q5. Apply Dijkstra's algorithm...",
        "char_count": 650,
        "token_estimate": 162,
        "difficulty_hint": "hard",
        "has_sub_parts": False,
        "marks_hint": 10,
        "year_hint": 2021,
    },
    {
        "chunk_id": "qna_004",
        "source_file": "CS301_PYQ_2023.pdf",
        "file_type": "pdf",
        "page_range": [1, 1],
        "chunk_type": "passage",
        "topic_cluster": 2,
        "topic_label": "asymptotic notation / Big-O",
        "questions": [
            "Define Big-O, Big-Omega, and Big-Theta notation."
        ],
        "answers": [],
        "raw_text": "Define Big-O, Big-Omega, and Big-Theta notation.",
        "char_count": 50,
        "token_estimate": 12,
        "difficulty_hint": "easy",
        "has_sub_parts": False,
        "marks_hint": 4,
        "year_hint": 2023,
    },
    {
        "chunk_id": "qna_005",
        "source_file": "CS301_PYQ_2020.pdf",
        "file_type": "pdf",
        "page_range": [8, 9],
        "chunk_type": "multi_qa",
        "topic_cluster": 1,
        "topic_label": "graph algorithms / BFS DFS",
        "questions": [
            "Trace BFS and DFS on a given graph starting from vertex 1.",
            "State the time complexity of each traversal.",
        ],
        "answers": [
            "BFS visits level by level using a queue.",
            "Both are O(V+E) for adjacency list representation.",
        ],
        "raw_text": "Q3. Trace BFS and DFS...",
        "char_count": 320,
        "token_estimate": 80,
        "difficulty_hint": "medium",
        "has_sub_parts": False,
        "marks_hint": 7,
        "year_hint": 2020,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Audio Pipeline Fixtures  (CourseChunk.to_dict() format)
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_CHUNKS: list[dict] = [
    {
        "chunk_id": "aud_sw_001",
        "course_id": "CS301",
        "course_name": "Algorithms & Data Structures",
        "lecture_id": "CS301_L07",
        "lecture_number": 7,
        "week_number": 4,
        "lecture_title": "Dynamic Programming – Part 1",
        "professor": "Dr. Sharma",
        "strategy": "sliding_window",
        "text": (
            "So dynamic programming is essentially about breaking a problem into subproblems "
            "and storing the results of those subproblems to avoid recomputation. "
            "The key insight is that if we have overlapping subproblems and optimal substructure, "
            "DP gives us an efficient solution. Let me walk through the Fibonacci example first..."
        ),
        "word_count": 62,
        "start_seconds": 0.0,
        "end_seconds": 75.0,
        "duration_seconds": 75.0,
        "segment_indices": [0, 1, 2, 3, 4, 5],
        "segment_count": 6,
        "concepts": ["dynamic programming", "overlapping subproblems", "optimal substructure"],
        "keywords": ["dynamic programming", "subproblems", "memoization", "fibonacci"],
        "is_concept_boundary": False,
        "concept_label": "dynamic programming",
        "avg_confidence": 0.92,
        "min_confidence": 0.87,
        "has_unclear_segments": False,
        "window_index": 0,
        "overlap_with_prev": 0,
        "overlap_with_next": 2,
        "source_file": "/lectures/CS301_L07.mp3",
        "file_name": "CS301_L07.mp3",
        "model": "gemini-2.5-flash",
    },
    {
        "chunk_id": "aud_cb_001",
        "course_id": "CS301",
        "course_name": "Algorithms & Data Structures",
        "lecture_id": "CS301_L07",
        "lecture_number": 7,
        "week_number": 4,
        "lecture_title": "Dynamic Programming – Part 1",
        "professor": "Dr. Sharma",
        "strategy": "concept_block",
        "text": (
            "Let us now look at memoization specifically. Memoization is a top-down approach "
            "where we store the result of every function call in a cache. "
            "If the same arguments appear again, we return the cached result. "
            "This converts exponential recursion to linear time for Fibonacci, for instance. "
            "The cache is typically a hash map keyed by the function arguments."
        ),
        "word_count": 75,
        "start_seconds": 210.0,
        "end_seconds": 310.0,
        "duration_seconds": 100.0,
        "segment_indices": [14, 15, 16, 17, 18, 19, 20, 21],
        "segment_count": 8,
        "concepts": ["memoization", "dynamic programming"],
        "keywords": ["memoization", "cache", "top-down", "fibonacci", "hash map"],
        "is_concept_boundary": True,
        "concept_label": "memoization",
        "avg_confidence": 0.89,
        "min_confidence": 0.82,
        "has_unclear_segments": False,
        "window_index": 1,
        "overlap_with_prev": 0,
        "overlap_with_next": 0,
        "source_file": "/lectures/CS301_L07.mp3",
        "file_name": "CS301_L07.mp3",
        "model": "gemini-2.5-flash",
    },
    {
        "chunk_id": "aud_ls_001",
        "course_id": "CS301",
        "course_name": "Algorithms & Data Structures",
        "lecture_id": "CS301_L07",
        "lecture_number": 7,
        "week_number": 4,
        "lecture_title": "Dynamic Programming – Part 1",
        "professor": "Dr. Sharma",
        "strategy": "lecture_summary",
        "text": (
            "Lecture summary: Introduction to dynamic programming covering optimal substructure, "
            "overlapping subproblems, memoization (top-down), and tabulation (bottom-up). "
            "Worked examples include Fibonacci, coin change, and longest common subsequence.\n\n"
            "Topics covered:\n- Dynamic programming\n- Memoization\n- Tabulation\n"
            "- Optimal substructure\n\nKey terms: DP, memoization, tabulation, subproblem, cache"
        ),
        "word_count": 85,
        "start_seconds": 0.0,
        "end_seconds": 3240.0,
        "duration_seconds": 3240.0,
        "segment_indices": list(range(200)),
        "segment_count": 200,
        "concepts": ["dynamic programming", "memoization", "tabulation", "optimal substructure"],
        "keywords": ["DP", "memoization", "tabulation", "subproblem", "cache"],
        "is_concept_boundary": True,
        "concept_label": "dynamic programming",
        "avg_confidence": 0.90,
        "min_confidence": 0.75,
        "has_unclear_segments": True,
        "window_index": 0,
        "overlap_with_prev": 0,
        "overlap_with_next": 0,
        "source_file": "/lectures/CS301_L07.mp3",
        "file_name": "CS301_L07.mp3",
        "model": "gemini-2.5-flash",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Document Pipeline Fixtures  (Document.metadata dict format)
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENT_CHUNKS: list[dict] = [
    {
        "chunk_id": "doc_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
        "source": "/docs/CLRS_Chapter15.pdf",
        "file_name": "CLRS_Chapter15.pdf",
        "file_type": "pdf",
        "course_id": "CS301",
        "chunk_index": 0,
        "total_chunks": 24,
        "page_start": 359,
        "page_end": 362,
        "section_title": "15.1 Rod Cutting",
        "element_types": "Title, NarrativeText, Table",
        "has_table": True,
        "has_image": False,
        "has_list": False,
        "table_count": 1,
        "image_count": 0,
        "list_item_count": 0,
        "image_summary_count": 0,
        "char_count": 2180,
        "word_count": 436,
        "text_preview": "The rod-cutting problem is the following. Given a rod of length n inches...",
        "keywords": "length, price, cutting, optimal, revenue, subproblem, recursive",
        "raw_text": (
            "The rod-cutting problem is the following. Given a rod of length n inches and a "
            "table of prices p_i for i=1,...,n, determine the maximum revenue r_n obtainable "
            "by cutting up the rod and selling the pieces. Note that if the price p_n for a "
            "rod of length n is large enough, an optimal solution may require no cutting at all."
        ),
        "tables_text": "Length | Price\n1 | 1\n2 | 5\n3 | 8\n4 | 9",
        "image_summaries": "[]",
        "ingested_at": "2024-09-01T10:00:00+00:00",
    },
    {
        "chunk_id": "doc_b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7",
        "source": "/docs/CLRS_Chapter15.pdf",
        "file_name": "CLRS_Chapter15.pdf",
        "file_type": "pdf",
        "course_id": "CS301",
        "chunk_index": 1,
        "total_chunks": 24,
        "page_start": 362,
        "page_end": 365,
        "section_title": "15.1 Rod Cutting",
        "element_types": "NarrativeText, CodeBlock",
        "has_table": False,
        "has_image": False,
        "has_list": False,
        "table_count": 0,
        "image_count": 0,
        "list_item_count": 0,
        "image_summary_count": 0,
        "char_count": 1850,
        "word_count": 370,
        "text_preview": "The following recursive procedure implements this straightforward top-down approach...",
        "keywords": "recursive, memoized, bottom, array, procedure, optimal, revenue",
        "raw_text": (
            "The following recursive procedure implements this straightforward top-down approach. "
            "MEMOIZED-CUT-ROD(p, n): let r[0..n] be a new array; for i=0 to n set r[i]=-∞. "
            "return MEMOIZED-CUT-ROD-AUX(p, n, r)."
        ),
        "tables_text": "",
        "image_summaries": "[]",
        "ingested_at": "2024-09-01T10:00:01+00:00",
    },
    {
        "chunk_id": "doc_c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
        "source": "/docs/lecture_slides_w4.pptx",
        "file_name": "lecture_slides_w4.pptx",
        "file_type": "pptx",
        "course_id": "CS301",
        "chunk_index": 0,
        "total_chunks": 8,
        "page_start": 1,
        "page_end": 3,
        "section_title": "Week 4: Dynamic Programming Overview",
        "element_types": "Title, NarrativeText, ListItem",
        "has_table": False,
        "has_image": True,
        "has_list": True,
        "table_count": 0,
        "image_count": 2,
        "list_item_count": 6,
        "image_summary_count": 2,
        "char_count": 980,
        "word_count": 196,
        "text_preview": "Dynamic Programming: Key Properties 1. Overlapping Subproblems...",
        "keywords": "subproblems, optimal, overlapping, structure, tabulation, memoization",
        "raw_text": "Dynamic Programming: Key Properties\n1. Overlapping Subproblems\n2. Optimal Substructure",
        "tables_text": "",
        "image_summaries": '[{"image_type": "diagram", "summary": "DP recursion tree showing overlapping subproblems"}]',
        "ingested_at": "2024-09-01T10:00:02+00:00",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Image Pipeline Fixtures  (Chunk.to_dict() format)
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_CHUNKS: list[dict] = [
    {
        "text": (
            "Week 4 — Dynamic Programming\n\n"
            "Overlapping Subproblems:\nFibonacci(5) calls Fibonacci(3) twice.\n"
            "This redundancy is solved by caching.\n\n"
            "Optimal Substructure:\nA shortest path from A to C through B means\n"
            "A→B and B→C are also shortest paths."
        ),
        "metadata": {
            "chunk_id": "img_0001abcd",
            "chunk_index": 0,
            "chunk_type": "prose",
            "source_file": "CS301_slides_scan_p04.pdf",
            "document_id": "doc_scan_001",
            "page_numbers": [4],
            "total_pages": 20,
            "course_id": "CS301",
            "course_name": "Algorithms & Data Structures",
            "subject_area": "Computer Science",
            "document_title": "Week 4 — Dynamic Programming",
            "section": "Overlapping Subproblems",
            "heading_path": ["Week 4 — Dynamic Programming", "Overlapping Subproblems"],
            "prev_chunk_id": "",
            "next_chunk_id": "img_0002abcd",
            "is_continuation": False,
            "part_index": 0,
            "token_count": 82,
            "char_count": 310,
            "word_count": 63,
            "line_count": 9,
            "ocr_confidence": 91.5,
            "image_type": "clean_scan",
            "skew_corrected": False,
            "skew_angle": 0.3,
            "ocr_warnings": [],
            "preprocessing_stages": ["upscale_x1.5", "binarize_otsu"],
            "has_code": False,
            "has_formula": False,
            "has_list": True,
            "is_definition": False,
            "is_example": True,
            "semantic_density": 0.71,
            "context_window": "Week 4 — Dynamic Programming\n\nOverlapping Subproblems:",
            "gemini_summary": "Slide explaining overlapping subproblems in DP using Fibonacci example.",
            "gemini_topic": "Dynamic Programming",
            "gemini_keywords": ["dynamic programming", "overlapping", "subproblems", "fibonacci"],
            "gemini_content_signals": ["educational", "has_example"],
            "created_at": "2024-09-01T11:00:00+00:00",
        },
    },
    {
        "text": (
            "Bottom-Up DP (Tabulation)\n\n"
            "Algorithm BOTTOM-UP-CUT-ROD(p, n):\n"
            "  let r[0..n] be a new array\n"
            "  r[0] = 0\n"
            "  for j = 1 to n:\n"
            "    q = -∞\n"
            "    for i = 1 to j:\n"
            "      q = max(q, p[i] + r[j-i])\n"
            "    r[j] = q\n"
            "  return r[n]"
        ),
        "metadata": {
            "chunk_id": "img_0002abcd",
            "chunk_index": 1,
            "chunk_type": "prose",
            "source_file": "CS301_slides_scan_p04.pdf",
            "document_id": "doc_scan_001",
            "page_numbers": [5],
            "total_pages": 20,
            "course_id": "CS301",
            "course_name": "Algorithms & Data Structures",
            "subject_area": "Computer Science",
            "document_title": "Week 4 — Dynamic Programming",
            "section": "Bottom-Up DP",
            "heading_path": ["Week 4 — Dynamic Programming", "Bottom-Up DP"],
            "prev_chunk_id": "img_0001abcd",
            "next_chunk_id": "",
            "is_continuation": False,
            "part_index": 1,
            "token_count": 95,
            "char_count": 270,
            "word_count": 48,
            "line_count": 12,
            "ocr_confidence": 87.2,
            "image_type": "clean_scan",
            "skew_corrected": True,
            "skew_angle": 1.2,
            "ocr_warnings": [],
            "preprocessing_stages": ["skew_correction_hough_1.20deg", "binarize_otsu"],
            "has_code": True,
            "has_formula": False,
            "has_list": False,
            "is_definition": False,
            "is_example": True,
            "semantic_density": 0.65,
            "context_window": "Bottom-Up DP (Tabulation)\n\nAlgorithm BOTTOM-UP-CUT-ROD",
            "gemini_summary": "Pseudocode for bottom-up tabulation approach to rod cutting problem.",
            "gemini_topic": "Tabulation / Bottom-Up DP",
            "gemini_keywords": ["tabulation", "bottom-up", "rod cutting", "pseudocode"],
            "gemini_content_signals": ["has_code", "pseudocode"],
            "created_at": "2024-09-01T11:00:01+00:00",
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Handwritten Pipeline Fixtures  (Chunk.metadata() dict format)
# ─────────────────────────────────────────────────────────────────────────────

HANDWRITTEN_CHUNKS: list[dict] = [
    {
        "course_id": "CS301",
        "course_name": "Algorithms & Data Structures",
        "subject": "Computer Science",
        "instructor": "Dr. Sharma",
        "semester": "Fall 2024",
        "university": "MIT",
        "tags": ["algorithms", "DP"],
        "source_file": "notes_lecture7_dp.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "topic": "Dynamic Programming – Memoization vs Tabulation",
        "content_type": "notes",
        "keywords": ["memoization", "tabulation", "DP", "recursion", "cache"],
        "summary": (
            "Handwritten notes comparing memoization and tabulation. "
            "Includes comparison table and worked Fibonacci example."
        ),
        "confidence": 0.88,
        "model_used": "gemini-2.5-flash",
        "has_diagrams": False,
        "has_equations": True,
        "has_tables": True,
        "ink_quality": "good",
        "writing_style": "print",
        "language": "en",
        "correction_notes": "",
        "text_preview": (
            "[CS301 | Algorithms & Data Structures | Dynamic Programming – Memoization vs Tabulation | Page 1]\n\n"
            "## Memoization vs Tabulation\n\n"
            "| Approach | Direction | Space | Implementation |\n"
            "| Memoization | Top-down | O(n) stack | Recursive |\n"
            "| Tabulation | Bottom-up | O(n) | Iterative |\n\n"
            "Fibonacci with memoization: T(n) = O(n), S(n) = O(n)"
        ),
    },
    {
        "course_id": "CS301",
        "course_name": "Algorithms & Data Structures",
        "subject": "Computer Science",
        "instructor": "Dr. Sharma",
        "semester": "Fall 2024",
        "university": "MIT",
        "tags": ["algorithms", "graphs"],
        "source_file": "notes_lecture9_graphs.pdf",
        "page_number": 2,
        "chunk_index": 0,
        "topic": "Dijkstra's Algorithm – Worked Example",
        "content_type": "mixed",
        "keywords": ["Dijkstra", "shortest path", "greedy", "priority queue"],
        "summary": "Worked example of Dijkstra's algorithm on a 5-node graph with edge weights.",
        "confidence": 0.74,
        "model_used": "gemini-2.5-pro",
        "has_diagrams": True,
        "has_equations": False,
        "has_tables": False,
        "ink_quality": "faded",
        "writing_style": "mixed",
        "language": "en",
        "correction_notes": "Node label at step 3 unclear: guessed 'C' vs 'G'",
        "text_preview": (
            "[CS301 | Algorithms & Data Structures | Dijkstra's Algorithm | Page 2]\n\n"
            "## Dijkstra's Algorithm\n\n"
            "[DIAGRAM: 5-node weighted graph, nodes A-E, edges with weights 4,2,7,1,3,5]\n\n"
            "Step 1: d(A)=0, all others = ∞. Min-heap: {A:0}\n"
            "Step 2: Extract A, relax A→B (d=4), A→C (d=2)\n"
            "Step 3: Extract C (d=2), relax C→D (d=3)"
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# YouTube Pipeline Fixtures  (TranscriptChunk as dict format)
# ─────────────────────────────────────────────────────────────────────────────

YOUTUBE_CHUNKS: list[dict] = [
    {
        "chunk_id": "yt_uuid_001",
        "video_id": "dQw4w9WgXcQ",  # placeholder
        "course_id": "CS301",
        "course_name": "Algorithms & Data Structures",
        "chunk_index": 0,
        "start_sec": 0.0,
        "end_sec": 182.4,
        "start_label": "00:00",
        "end_label": "03:02",
        "raw_text": (
            "Welcome everyone. Today we are going to talk about dynamic programming. "
            "Dynamic programming is a method for solving complex problems by breaking them down "
            "into simpler subproblems. It is applicable when the problem has overlapping subproblems "
            "and optimal substructure properties. The key idea is to avoid computing the same "
            "subproblem twice by storing the results."
        ),
        "summary": "Introduction to dynamic programming covering overlapping subproblems...",
        "topic": "Introduction to Dynamic Programming",
        "concept_tags": ["dynamic", "programming", "subproblems", "optimal", "storing"],
        "chapter_title": "Introduction",
        "chapter_index": 0,
        "video_title": "Dynamic Programming Explained – MIT OCW Style",
        "author": "AlgoProf",
        "channel_url": "https://www.youtube.com/c/AlgoProf",
        "duration_sec": 3600,
        "publish_date": "2023-06-15",
        "thumbnail_url": "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
        "yt_keywords": ["algorithms", "dynamic programming", "computer science"],
        "is_generated_transcript": False,
        "transcript_language": "en",
        "instructor": "Prof. Mehta",
        "semester": "Fall 2024",
        "subject": "Computer Science",
        "tags": ["algorithms"],
        "deep_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0s",
    },
    {
        "chunk_id": "yt_uuid_002",
        "video_id": "dQw4w9WgXcQ",
        "course_id": "CS301",
        "course_name": "Algorithms & Data Structures",
        "chunk_index": 3,
        "start_sec": 540.0,
        "end_sec": 720.0,
        "start_label": "09:00",
        "end_label": "12:00",
        "raw_text": (
            "Now let us look at the Fibonacci sequence as our first example. "
            "If we compute F(5) naively, we end up calling F(3) twice, F(2) three times, "
            "and F(1) five times. This exponential blowup is why we need memoization. "
            "With memoization we store each computed F(n) in a table. "
            "The first time we compute F(3) we store it, and every subsequent call just reads it."
        ),
        "summary": "Fibonacci example demonstrating exponential blowup and memoization fix...",
        "topic": "Fibonacci Example and Memoization",
        "concept_tags": ["fibonacci", "memoization", "exponential", "table", "compute"],
        "chapter_title": "Worked Example: Fibonacci",
        "chapter_index": 2,
        "video_title": "Dynamic Programming Explained – MIT OCW Style",
        "author": "AlgoProf",
        "channel_url": "https://www.youtube.com/c/AlgoProf",
        "duration_sec": 3600,
        "publish_date": "2023-06-15",
        "thumbnail_url": "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
        "yt_keywords": ["algorithms", "dynamic programming", "computer science"],
        "is_generated_transcript": False,
        "transcript_language": "en",
        "instructor": "Prof. Mehta",
        "semester": "Fall 2024",
        "subject": "Computer Science",
        "tags": ["algorithms"],
        "deep_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=540s",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Scenarios  (ground-truth annotated test cases)
# ─────────────────────────────────────────────────────────────────────────────

# Each scenario maps to what retrieve() would return for a given query.
# chunk dicts below follow RetrievedChunk.to_dict() format.

def _make_retrieved(chunk_id, source_key, score, topic="", text="", **kwargs):
    base = {
        "chunk_id": chunk_id,
        "source_key": source_key,
        "score": score,
        "rrf_score": score,
        "retrieval_methods": ["dense"],
        "citation_label": f"{source_key}:{chunk_id}",
        "display_text": text or f"[{source_key}] sample text for {chunk_id}",
        "source_file": kwargs.get("source_file", "test_file.pdf"),
        "course_id": "CS301",
        "page_range": kwargs.get("page_range", "1"),
        "section_title": kwargs.get("section_title", ""),
        "topic": topic,
        "chunk_type": kwargs.get("chunk_type", ""),
        "keywords": kwargs.get("keywords", []),
        "year_hint": kwargs.get("year_hint"),
        "marks_hint": kwargs.get("marks_hint"),
        "image_type": kwargs.get("image_type", ""),
        "image_summary": kwargs.get("image_summary", ""),
        "start_label": kwargs.get("start_label", ""),
        "lecture_title": kwargs.get("lecture_title", ""),
        "professor": kwargs.get("professor", ""),
        "video_title": kwargs.get("video_title", ""),
        "deep_link": kwargs.get("deep_link", ""),
        "chapter_title": kwargs.get("chapter_title", ""),
        "has_equations": kwargs.get("has_equations", False),
        "has_diagrams": kwargs.get("has_diagrams", False),
        "has_table": kwargs.get("has_table", False),
        "difficulty": kwargs.get("difficulty", ""),
    }
    return base


RETRIEVAL_SCENARIOS: list[dict] = [
    # ── Scenario 1: Standard ASK query on DP ─────────────────────────────────
    {
        "id": "scen_ask_dp_001",
        "query": "What is dynamic programming and how does memoization work?",
        "mode": "ask",
        "course_id": "CS301",
        "relevant_chunk_ids": {"qna_001", "qna_002", "aud_cb_001", "doc_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"},
        "expected_sources": {"qna", "audio", "documents"},
        "excluded_sources": set(),
        "k": 8,
        "relevance_grades": {
            "qna_001": 2,
            "qna_002": 2,
            "aud_cb_001": 2,
            "doc_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6": 1,
            "aud_sw_001": 1,
        },
        "retrieved_chunks": [
            _make_retrieved("qna_001",   "qna",       0.0420, topic="Dynamic Programming", year_hint=2022, marks_hint=8, chunk_type="single_qa"),
            _make_retrieved("aud_cb_001","audio",     0.0388, topic="Memoization", start_label="03:30"),
            _make_retrieved("qna_002",   "qna",       0.0352, topic="Memoization vs Tabulation", year_hint=2022, marks_hint=6, chunk_type="multi_qa"),
            _make_retrieved("doc_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", "documents", 0.0301, topic="Rod Cutting", has_table=True),
            _make_retrieved("aud_sw_001","audio",     0.0280, topic="DP Introduction"),
            _make_retrieved("yt_uuid_002","youtube",  0.0245, topic="Fibonacci Example", video_title="DP Explained", chapter_title="Worked Example"),
            _make_retrieved("img_0001abcd","image",   0.0210, topic="DP Diagram"),
            _make_retrieved("hw_001",    "handwritten",0.0180, topic="Memoization Notes"),
        ],
        "planner_data": {
            "source_budgets": [
                {"source_key": "qna",       "dense_k": 6, "bm25_k": 4, "use_mmr": False, "mmr_k": 5, "mmr_lambda": 0.6, "priority_weight": 2.0},
                {"source_key": "audio",     "dense_k": 8, "bm25_k": 0, "use_mmr": True,  "mmr_k": 5, "mmr_lambda": 0.6, "priority_weight": 1.8},
                {"source_key": "documents", "dense_k": 6, "bm25_k": 3, "use_mmr": True,  "mmr_k": 4, "mmr_lambda": 0.6, "priority_weight": 2.2},
            ],
            "sub_queries": [
                {"text": "define dynamic programming optimal substructure overlapping subproblems", "facet": "definition"},
                {"text": "how does memoization cache subproblem results", "facet": "mechanism"},
                {"text": "dynamic programming memoization tabulation comparison", "facet": "comparison"},
            ],
            "key_concepts": ["dynamic programming", "memoization", "optimal substructure"],
            "domain_keywords": ["dynamic programming", "memoization", "subproblems", "cache", "overlapping"],
            "expected_answer_format": "structured_explanation",
            "expected_answer_length": "medium",
            "high_value_signals": ["contains definition", "has worked example"],
            "low_value_signals": ["administrative content"],
        },
        "sample_answer": (
            "## Dynamic Programming\n\n"
            "Dynamic programming (DP) is an algorithmic technique for solving problems with "
            "**overlapping subproblems** and **optimal substructure** [1]. Instead of recomputing "
            "the same subproblem repeatedly, DP stores results for reuse [2].\n\n"
            "## Memoization\n\n"
            "Memoization is the top-down DP approach [3]. When a function is called, its result "
            "is stored in a hash map keyed by the arguments. Subsequent calls with the same "
            "arguments return the cached result in O(1) [2]. For the **Fibonacci** sequence, "
            "this reduces time complexity from O(2^n) to O(n) [5].\n\n"
            "## Key Difference from Tabulation\n\n"
            "Tabulation is the bottom-up approach — it fills an array iteratively without recursion [3][4]."
        ),
        "original_sub_query_count": 3,
        "deduplicated_sub_query_count": 3,
        "planner_ms": 1200.0,
        "total_ms": 3800.0,
    },

    # ── Scenario 2: QUIZ mode on graph algorithms ─────────────────────────────
    {
        "id": "scen_quiz_graphs_001",
        "query": "Generate quiz questions on Dijkstra's algorithm",
        "mode": "quiz",
        "course_id": "CS301",
        "relevant_chunk_ids": {"qna_003", "qna_005", "hw_002"},
        "expected_sources": {"qna", "handwritten", "documents"},
        "excluded_sources": {"youtube"},
        "k": 8,
        "relevance_grades": {
            "qna_003": 2,
            "qna_005": 1,
            "hw_002": 1,
        },
        "retrieved_chunks": [
            _make_retrieved("qna_003",    "qna",        0.0451, topic="Dijkstra", year_hint=2021, marks_hint=10, chunk_type="single_qa", difficulty="hard"),
            _make_retrieved("qna_005",    "qna",        0.0390, topic="Graph Traversal", chunk_type="multi_qa", difficulty="medium"),
            _make_retrieved("hw_002",     "handwritten",0.0320, topic="Dijkstra worked example", has_diagrams=True),
            _make_retrieved("doc_diag_1", "documents",  0.0298, topic="Graph Algorithms overview"),
            _make_retrieved("img_graph_1","image",      0.0241, topic="Graph diagram", image_type="diagram"),
            _make_retrieved("aud_graph_1","audio",      0.0182, topic="Graph lecture"),
            _make_retrieved("qna_004",    "qna",        0.0161, topic="Asymptotic notation", difficulty="easy"),
            _make_retrieved("yt_uuid_003","youtube",    0.0120, topic="Graph theory intro"),
        ],
        "planner_data": {
            "source_budgets": [
                {"source_key": "qna",        "dense_k": 9, "bm25_k": 5, "use_mmr": False, "mmr_k": 5, "mmr_lambda": 0.6, "priority_weight": 2.8},
                {"source_key": "handwritten","dense_k": 4, "bm25_k": 0, "use_mmr": False, "mmr_k": 3, "mmr_lambda": 0.6, "priority_weight": 1.5},
                {"source_key": "documents",  "dense_k": 5, "bm25_k": 2, "use_mmr": True,  "mmr_k": 4, "mmr_lambda": 0.6, "priority_weight": 1.8},
            ],
            "quiz_config": {
                "question_types": ["mcq", "short_answer", "fill_in_blank"],
                "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
                "num_questions": 5,
                "include_answer_key": True,
                "focus_on_exam_patterns": True,
            },
            "sub_queries": [
                {"text": "Dijkstra's algorithm shortest path exam questions", "facet": "exam_phrasing"},
                {"text": "how does Dijkstra's algorithm work step by step", "facet": "mechanism"},
            ],
            "key_concepts": ["Dijkstra", "shortest path", "greedy", "priority queue"],
            "domain_keywords": ["Dijkstra", "shortest", "greedy", "priority", "relaxation"],
            "expected_answer_format": "quiz_questions",
            "expected_answer_length": "long",
            "high_value_signals": ["contains exam question", "has worked example"],
            "low_value_signals": ["administrative content", "unrelated subtopic"],
        },
        "sample_answer": (
            "1. (MCQ) What is the time complexity of Dijkstra's algorithm with a binary heap? [2]\n"
            "   a) O(V^2)  b) O((V+E) log V)  c) O(E log V)  d) O(V log V)\n"
            "   **Answer: b** — Using a min-heap, each vertex extraction is O(log V) and "
            "each edge relaxation is O(log V), giving O((V+E) log V) total. [1]\n\n"
            "2. (Short Answer) Apply Dijkstra's algorithm to find the shortest path from A to F "
            "in a weighted graph. Show all steps. [10 marks] [1]\n"
            "   **Answer key**: Initialize d(A)=0, all others ∞. Extract minimum, relax edges..."
        ),
        "original_sub_query_count": 3,
        "deduplicated_sub_query_count": 2,
        "planner_ms": 1450.0,
        "total_ms": 4200.0,
    },

    # ── Scenario 3: SUMMARIZE mode ────────────────────────────────────────────
    {
        "id": "scen_summarize_dp_001",
        "query": "Summarize the key topics in dynamic programming for this course",
        "mode": "summarize",
        "course_id": "CS301",
        "relevant_chunk_ids": {"qna_001", "qna_002", "aud_ls_001", "doc_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", "yt_uuid_001"},
        "expected_sources": {"qna", "audio", "documents", "youtube"},
        "excluded_sources": set(),
        "k": 8,
        "relevance_grades": {
            "qna_001": 1,
            "qna_002": 1,
            "aud_ls_001": 2,
            "doc_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6": 2,
            "yt_uuid_001": 1,
        },
        "retrieved_chunks": [
            _make_retrieved("aud_ls_001", "audio",     0.0462, topic="Lecture Summary: DP"),
            _make_retrieved("doc_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", "documents", 0.0401, has_table=True),
            _make_retrieved("yt_uuid_001","youtube",   0.0358, topic="DP Introduction", chapter_title="Introduction"),
            _make_retrieved("qna_001",    "qna",       0.0310, topic="DP Optimal Substructure"),
            _make_retrieved("qna_002",    "qna",       0.0285, topic="Memoization vs Tabulation"),
            _make_retrieved("img_0001abcd","image",    0.0241, topic="DP slide"),
            _make_retrieved("hw_001",     "handwritten",0.0198, topic="Memoization Notes"),
            _make_retrieved("doc_b_chunk","documents", 0.0170, topic="Rod Cutting Code"),
        ],
        "planner_data": {
            "source_budgets": [
                {"source_key": "documents", "dense_k": 7, "bm25_k": 3, "use_mmr": True,  "mmr_k": 5, "mmr_lambda": 0.6, "priority_weight": 2.2},
                {"source_key": "audio",     "dense_k": 8, "bm25_k": 0, "use_mmr": True,  "mmr_k": 5, "mmr_lambda": 0.6, "priority_weight": 2.0},
                {"source_key": "youtube",   "dense_k": 6, "bm25_k": 0, "use_mmr": True,  "mmr_k": 4, "mmr_lambda": 0.6, "priority_weight": 1.8},
                {"source_key": "qna",       "dense_k": 5, "bm25_k": 2, "use_mmr": False, "mmr_k": 4, "mmr_lambda": 0.6, "priority_weight": 1.5},
            ],
            "summary_config": {
                "depth": "standard",
                "include_key_terms": True,
                "include_examples": True,
                "format": "structured_headers",
            },
            "sub_queries": [
                {"text": "dynamic programming course overview key topics", "facet": "topic_overview"},
                {"text": "memoization tabulation DP approaches comparison", "facet": "comparison"},
                {"text": "dynamic programming applications rod cutting LCS", "facet": "application"},
            ],
            "key_concepts": ["dynamic programming", "memoization", "tabulation", "optimal substructure"],
            "domain_keywords": ["DP", "memoization", "tabulation", "overlapping", "substructure", "cache"],
            "expected_answer_format": "topic_summary",
            "expected_answer_length": "long",
            "high_value_signals": ["lecture summary", "covers multiple topics", "has examples"],
            "low_value_signals": ["only tangential mention"],
        },
        "sample_answer": (
            "## Dynamic Programming: Course Summary\n\n"
            "### Core Concepts\n\n"
            "Dynamic programming solves problems with **overlapping subproblems** and "
            "**optimal substructure** [1][4]. The lecture covered two primary approaches [3]:\n\n"
            "- **Memoization** (top-down): cache results of recursive calls [5]\n"
            "- **Tabulation** (bottom-up): fill a table iteratively [2]\n\n"
            "### Key Applications\n\n"
            "The rod-cutting problem [2] and Fibonacci sequence [3] were core examples. "
            "LCS and coin change were also mentioned in the lecture summary [1].\n\n"
            "### Key Terms\n\n"
            "DP, memoization, tabulation, overlapping subproblems, optimal substructure, cache"
        ),
        "original_sub_query_count": 4,
        "deduplicated_sub_query_count": 3,
        "planner_ms": 1380.0,
        "total_ms": 4900.0,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: all chunks as one flat retrieval pool
# ─────────────────────────────────────────────────────────────────────────────

SYLLABUS_TOPICS = [
    "Dynamic Programming",
    "Greedy Algorithms",
    "Graph Algorithms",
    "Sorting and Searching",
    "Asymptotic Analysis",
    "Divide and Conquer",
    "Shortest Paths",
    "Minimum Spanning Trees",
]