"""
rate_limiting package
=====================
Token-based Redis rate limiter for the CourseGPT API.

Two-phase usage:
  1. check_token_budget()  — pre-flight check before calling Gemini
  2. record_token_usage()  — write actual token count after Gemini responds
"""
from .rate_limiter import (
    check_token_budget,
    record_token_usage,
    get_user_usage,
    reset_user_limits,
    TokenRateLimitResult,
)

__all__ = [
    "check_token_budget",
    "record_token_usage",
    "get_user_usage",
    "reset_user_limits",
    "TokenRateLimitResult",
]
