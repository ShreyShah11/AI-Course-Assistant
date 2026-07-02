"""
rate_limiter.py  —  Token-based Redis Rate Limiter
====================================================
Instead of counting requests, every check operates in **tokens**:
the actual token count returned by Gemini's usage_metadata.total_token_count.

How it works
------------
Each retrieval request is a two-phase operation:

  Phase 1 — Pre-flight check  (BEFORE calling Gemini)
    Read current token usage across all 3 tiers. If any tier is already
    at/above its limit, reject immediately with 429 before any API cost.

  Phase 2 — Record actual usage  (AFTER Gemini responds)
    Write the exact token count from usage_metadata into Redis. The next
    pre-flight check will see the real accumulated cost.
                                                    
Three tiers (all checked atomically via Lua — one Redis RTT)
-------------------------------------------------------------
  ┌────────────────────────────────────────────────────────┐
  │ Tier 1: Sliding Window (per-minute token budget)       │
  │   Sorted set where each member encodes a token count.  │
  │   Sum of all tokens within the 60-sec window must be   │
  │   below TOKEN_WINDOW_LIMIT.                            │
  │   No boundary spike: window is truly sliding.          │
  ├────────────────────────────────────────────────────────┤
  │ Tier 2: Token Bucket (burst smoothing)                 │
  │   Bucket holds up to BUCKET_CAPACITY tokens.           │
  │   Refills at BUCKET_REFILL_RATE tokens/sec.            │
  │   Each request deducts its actual token count.         │
  │   Prevents microsecond-scale bursts.                   │
  ├────────────────────────────────────────────────────────┤
  │ Tier 3: Daily Quota (absolute ceiling)                 │
  │   Simple counter of total tokens today (UTC).          │
  │   Resets at midnight UTC.                              │
  └────────────────────────────────────────────────────────┘

Redis key layout
----------------
  rl_tok:{user_id}:{mode}:sw          Sorted set — sliding window entries
                                       score=timestamp_µs, member="{ts}:{tokens}"
  rl_tok:{user_id}:bucket             Hash — {tokens: float, last_refill: float}
  rl_tok:{user_id}:daily:{YYYY-MM-DD} String — total tokens consumed today

Response headers
----------------
  X-RateLimit-Tokens-Window-Limit     : per-minute token budget
  X-RateLimit-Tokens-Window-Used      : tokens used in current 60-sec window
  X-RateLimit-Tokens-Window-Remaining : tokens left in current window
  X-RateLimit-Tokens-Daily-Limit      : daily token budget
  X-RateLimit-Tokens-Daily-Used       : tokens consumed today
  X-RateLimit-Tokens-Daily-Remaining  : tokens left today
  X-RateLimit-Bucket-Tokens           : current token bucket level
  Retry-After                         : seconds to wait (429 only)

Environment variables
---------------------
  RATE_LIMIT_ENABLED                   bool   (default: true)

  # Tier 1: Sliding window (tokens per 60-sec window)
  RATE_LIMIT_STUDENT_TOKENS_PER_MIN    int    (default: 50000)
  RATE_LIMIT_TEACHER_TOKENS_PER_MIN    int    (default: 150000)

  # Tier 2: Token bucket
  RATE_LIMIT_STUDENT_BUCKET_CAPACITY   int    (default: 80000)
  RATE_LIMIT_TEACHER_BUCKET_CAPACITY   int    (default: 250000)
  RATE_LIMIT_STUDENT_BUCKET_REFILL     float  (default: 800.0  tokens/sec)
  RATE_LIMIT_TEACHER_BUCKET_REFILL     float  (default: 2500.0 tokens/sec)

  # Tier 3: Daily quota (total tokens per UTC day)
  RATE_LIMIT_STUDENT_TOKENS_DAILY      int    (default: 500000)
  RATE_LIMIT_TEACHER_TOKENS_DAILY      int    (default: 2000000)

  REDIS_URL                            str    (default: redis://localhost:6379/0)
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes")

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)

# Tier 1 — Sliding window
STUDENT_TOKENS_PER_MIN = _env_int("RATE_LIMIT_STUDENT_TOKENS_PER_MIN", 50_000)
TEACHER_TOKENS_PER_MIN = _env_int("RATE_LIMIT_TEACHER_TOKENS_PER_MIN", 150_000)

# Tier 2 — Token bucket (role-specific capacity + refill)
STUDENT_BUCKET_CAPACITY = _env_int  ("RATE_LIMIT_STUDENT_BUCKET_CAPACITY", 80_000)
TEACHER_BUCKET_CAPACITY = _env_int  ("RATE_LIMIT_TEACHER_BUCKET_CAPACITY", 250_000)
STUDENT_BUCKET_REFILL   = _env_float("RATE_LIMIT_STUDENT_BUCKET_REFILL",   800.0)
TEACHER_BUCKET_REFILL   = _env_float("RATE_LIMIT_TEACHER_BUCKET_REFILL",  2_500.0)

# Tier 3 — Daily quota
STUDENT_TOKENS_DAILY = _env_int("RATE_LIMIT_STUDENT_TOKENS_DAILY",   500_000)
TEACHER_TOKENS_DAILY = _env_int("RATE_LIMIT_TEACHER_TOKENS_DAILY", 2_000_000)

# Redis key prefix
_PREFIX = "rl_tok"


# ─────────────────────────────────────────────────────────────────────────────
# Redis connection (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is None:
        from redis import Redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = Redis.from_url(url, decode_responses=True)
    return _redis_client


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenRateLimitResult:
    """Result of a token-based rate-limit pre-flight check."""
    allowed: bool
    rejected_by: str = ""  # "sliding_window" | "token_bucket" | "daily_quota" | ""

    # Window state
    window_limit:     int   = 0
    window_used:      int   = 0
    window_remaining: int   = 0

    # Bucket state
    bucket_tokens:   float = 0.0
    bucket_capacity: int   = 0

    # Daily state
    daily_limit:     int   = 0
    daily_used:      int   = 0
    daily_remaining: int   = 0

    # Retry guidance (populated on rejection)
    retry_after_secs: float = 0.0

    @property
    def headers(self) -> dict[str, str]:
        """RFC-inspired token-rate-limit response headers."""
        h = {
            "X-RateLimit-Tokens-Window-Limit":     str(self.window_limit),
            "X-RateLimit-Tokens-Window-Used":      str(self.window_used),
            "X-RateLimit-Tokens-Window-Remaining": str(max(0, self.window_remaining)),
            "X-RateLimit-Tokens-Daily-Limit":      str(self.daily_limit),
            "X-RateLimit-Tokens-Daily-Used":       str(self.daily_used),
            "X-RateLimit-Tokens-Daily-Remaining":  str(max(0, self.daily_remaining)),
            "X-RateLimit-Bucket-Tokens":           str(int(self.bucket_tokens)),
        }
        if not self.allowed:
            h["Retry-After"] = str(math.ceil(self.retry_after_secs))
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Lua scripts (atomic Redis operations)
# ─────────────────────────────────────────────────────────────────────────────
#
# Two separate scripts:
#   _LUA_PRECHECK  — read-only inspection (+ evict expired window entries)
#   _LUA_RECORD    — write actual token usage after Gemini responds
#
# Separating them lets us handle the case where Gemini fails (we don't record
# usage for a failed call) while still doing the pre-flight check correctly.

# ── Pre-flight check ─────────────────────────────────────────────────────────
# KEYS[1] = sliding window sorted set  (rl_tok:{user_id}:{mode}:sw)
# KEYS[2] = token bucket hash          (rl_tok:{user_id}:bucket)
# KEYS[3] = daily counter              (rl_tok:{user_id}:daily:{date})
#
# ARGV[1] = now_us          (int: current time in microseconds)
# ARGV[2] = win_secs        (int: window width, always 60)
# ARGV[3] = win_token_limit (int: max tokens allowed in window)
# ARGV[4] = bkt_cap         (int: token bucket capacity)
# ARGV[5] = bkt_rate        (float: tokens refilled per second)
# ARGV[6] = daily_limit     (int: max tokens per day)
#
# Returns: {allowed, window_tokens, bucket_tokens, daily_tokens, rejection_code}
#   rejection_code: 0=allowed, 1=window, 2=bucket, 3=daily
_LUA_PRECHECK = """
local sw_key    = KEYS[1]
local bkt_key   = KEYS[2]
local day_key   = KEYS[3]

local now_us    = tonumber(ARGV[1])
local win_secs  = tonumber(ARGV[2])
local win_limit = tonumber(ARGV[3])
local bkt_cap   = tonumber(ARGV[4])
local bkt_rate  = tonumber(ARGV[5])
local day_limit = tonumber(ARGV[6])

local now_s     = now_us / 1000000.0
local cutoff_us = now_us - (win_secs * 1000000)

-- ═══ Tier 1: Sliding window — evict expired, sum remaining token costs ═══
redis.call('ZREMRANGEBYSCORE', sw_key, '-inf', cutoff_us)

local entries = redis.call('ZRANGE', sw_key, 0, -1)
local win_tokens = 0
for _, member in ipairs(entries) do
    local colon = string.find(member, ':', 1, true)
    if colon then
        local t = tonumber(string.sub(member, colon + 1)) or 0
        win_tokens = win_tokens + t
    end
end

if win_tokens >= win_limit then
    return {0, win_tokens, -1, -1, 1}   -- rejected: window
end

-- ═══ Tier 2: Token bucket — compute current available tokens ═══
local bkt_tokens  = tonumber(redis.call('HGET', bkt_key, 'tokens'))
local bkt_last    = tonumber(redis.call('HGET', bkt_key, 'last_refill'))
if bkt_tokens == nil then bkt_tokens = bkt_cap end
if bkt_last   == nil then bkt_last   = now_s   end

local elapsed     = math.max(0, now_s - bkt_last)
local available   = math.min(bkt_cap, bkt_tokens + elapsed * bkt_rate)

if available < 1 then
    return {0, win_tokens, available, -1, 2}  -- rejected: bucket empty
end

-- ═══ Tier 3: Daily quota ═══
local day_used = tonumber(redis.call('GET', day_key)) or 0
if day_used >= day_limit then
    return {0, win_tokens, available, day_used, 3}  -- rejected: daily quota
end

-- ═══ All tiers passed ═══
return {1, win_tokens, available, day_used, 0}
"""

# ── Record actual usage ───────────────────────────────────────────────────────
# KEYS[1] = sliding window sorted set
# KEYS[2] = token bucket hash
# KEYS[3] = daily counter
#
# ARGV[1] = now_us        (current time in microseconds)
# ARGV[2] = win_secs      (window TTL for EXPIRE)
# ARGV[3] = actual_tokens (int: real token count from Gemini usage_metadata)
# ARGV[4] = bkt_cap       (int: bucket capacity)
# ARGV[5] = bkt_rate      (float: refill rate tokens/sec)
# ARGV[6] = daily_ttl     (int: seconds until midnight UTC)
#
# Returns: {new_bucket_tokens, new_daily_total}
_LUA_RECORD = """
local sw_key    = KEYS[1]
local bkt_key   = KEYS[2]
local day_key   = KEYS[3]

local now_us    = tonumber(ARGV[1])
local win_secs  = tonumber(ARGV[2])
local tokens    = tonumber(ARGV[3])
local bkt_cap   = tonumber(ARGV[4])
local bkt_rate  = tonumber(ARGV[5])
local day_ttl   = tonumber(ARGV[6])

local now_s     = now_us / 1000000.0

-- ═══ Tier 1: Add this call to the sliding window ═══
-- member format: "{timestamp_us}:{tokens}" — encodes token cost in the member
local member = tostring(now_us) .. ':' .. tostring(tokens)
redis.call('ZADD', sw_key, now_us, member)
redis.call('EXPIRE', sw_key, win_secs + 10)

-- ═══ Tier 2: Deduct actual tokens from bucket (refill first) ═══
local bkt_tokens = tonumber(redis.call('HGET', bkt_key, 'tokens'))
local bkt_last   = tonumber(redis.call('HGET', bkt_key, 'last_refill'))
if bkt_tokens == nil then bkt_tokens = bkt_cap end
if bkt_last   == nil then bkt_last   = now_s   end

local elapsed    = math.max(0, now_s - bkt_last)
local refilled   = math.min(bkt_cap, bkt_tokens + elapsed * bkt_rate)
local remaining  = math.max(0, refilled - tokens)   -- clamp: never go negative

redis.call('HSET', bkt_key, 'tokens', tostring(remaining), 'last_refill', tostring(now_s))
redis.call('EXPIRE', bkt_key, 7200)   -- 2-hour TTL; bucket refills naturally

-- ═══ Tier 3: Increment daily token counter ═══
local new_daily = redis.call('INCRBY', day_key, tokens)
if tonumber(new_daily) == tokens then   -- first write today → set TTL
    redis.call('EXPIRE', day_key, day_ttl)
end

return {remaining, new_daily}
"""

# SHA cache for both scripts
_precheck_sha: Optional[str] = None
_record_sha:   Optional[str] = None


def _load_scripts(rc) -> tuple[str, str]:
    global _precheck_sha, _record_sha
    if _precheck_sha is None:
        _precheck_sha = rc.script_load(_LUA_PRECHECK)
    if _record_sha is None:
        _record_sha = rc.script_load(_LUA_RECORD)
    return _precheck_sha, _record_sha


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _thresholds(role: str) -> tuple[int, int, int, float, int]:
    """Return (win_limit, bkt_cap, bkt_refill, daily_limit) for a role."""
    if role == "teacher":
        return (
            TEACHER_TOKENS_PER_MIN,
            TEACHER_BUCKET_CAPACITY,
            TEACHER_BUCKET_REFILL,
            TEACHER_TOKENS_DAILY,
        )
    return (
        STUDENT_TOKENS_PER_MIN,
        STUDENT_BUCKET_CAPACITY,
        STUDENT_BUCKET_REFILL,
        STUDENT_TOKENS_DAILY,
    )


def _daily_ttl_secs() -> int:
    """Seconds remaining until midnight UTC."""
    now      = time.time()
    midnight = (int(now) // 86_400 + 1) * 86_400
    return max(1, int(midnight - now) + 1)


def _redis_keys(user_id: str, mode: str) -> tuple[str, str, str]:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    return (
        f"{_PREFIX}:{user_id}:{mode}:sw",          # sliding window
        f"{_PREFIX}:{user_id}:bucket",              # token bucket
        f"{_PREFIX}:{user_id}:daily:{today}",       # daily counter
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check_token_budget(
    user_id: str,
    role:    str,   # "student" | "teacher"
    mode:    str,   # "ask" | "quiz" | "summarize"
) -> TokenRateLimitResult:
    """
    Phase 1 — Pre-flight check (BEFORE calling Gemini).

    Reads current token usage across all 3 tiers. Rejects immediately if any
    tier is already at/above its limit, saving the API cost of a rejected user.

    Returns TokenRateLimitResult. Caller must check .allowed and raise 429
    if False. On success, save the result — it carries the pre-call state
    which becomes the rate-limit headers on the response.

    This is intentionally optimistic: we don't deduct any tokens here.
    Actual deduction happens in record_token_usage() after Gemini responds.
    """
    if not RATE_LIMIT_ENABLED:
        win_limit, bkt_cap, _, daily_limit = _thresholds(role)
        return TokenRateLimitResult(
            allowed=True,
            window_limit=win_limit,   window_used=0, window_remaining=win_limit,
            bucket_tokens=float(bkt_cap), bucket_capacity=bkt_cap,
            daily_limit=daily_limit,  daily_used=0,  daily_remaining=daily_limit,
        )

    win_limit, bkt_cap, bkt_rate, daily_limit = _thresholds(role)
    now_us   = int(time.time() * 1_000_000)
    sw_key, bkt_key, day_key = _redis_keys(user_id, mode)

    try:
        rc = _get_redis()
        precheck_sha, _ = _load_scripts(rc)

        raw = rc.evalsha(
            precheck_sha,
            3,                    # KEYS count
            sw_key, bkt_key, day_key,
            str(now_us),          # ARGV[1]
            "60",                 # ARGV[2] window width
            str(win_limit),       # ARGV[3]
            str(bkt_cap),         # ARGV[4]
            str(bkt_rate),        # ARGV[5]
            str(daily_limit),     # ARGV[6]
        )

        # raw = [allowed, window_tokens, bucket_tokens, daily_tokens, rejection_code]
        allowed      = int(raw[0]) == 1
        win_used     = int(raw[1])
        bkt_avail    = float(raw[2]) if raw[2] is not None and float(raw[2]) >= 0 else 0.0
        day_used     = int(raw[3])   if raw[3] is not None and int(raw[3])   >= 0 else 0
        reject_code  = int(raw[4])

        # Map rejection codes → human-readable reasons + retry hints
        rejected_by   = ""
        retry_after   = 0.0
        if not allowed:
            if reject_code == 1:
                rejected_by = "sliding_window"
                # Estimate: oldest entry will age out in at most 60 sec
                retry_after = max(5.0, 60.0 * (win_used - win_limit) / max(1, win_used))
            elif reject_code == 2:
                rejected_by = "token_bucket"
                # Time to refill enough for a minimal request (~500 tokens)
                min_tokens  = 500
                retry_after = max(1.0, (min_tokens - bkt_avail) / max(0.001, bkt_rate))
            elif reject_code == 3:
                rejected_by = "daily_quota"
                retry_after = float(_daily_ttl_secs())

        result = TokenRateLimitResult(
            allowed          = allowed,
            rejected_by      = rejected_by,
            window_limit     = win_limit,
            window_used      = win_used,
            window_remaining = max(0, win_limit - win_used),
            bucket_tokens    = bkt_avail,
            bucket_capacity  = bkt_cap,
            daily_limit      = daily_limit,
            daily_used       = day_used,
            daily_remaining  = max(0, daily_limit - day_used),
            retry_after_secs = retry_after,
        )

        if not allowed:
            logger.warning(
                "Token rate limit REJECTED  user=%s role=%s mode=%s tier=%s "
                "win=%d/%d  bucket=%.0f/%d  daily=%d/%d  retry=%.0fs",
                user_id[:8], role, mode, rejected_by,
                win_used, win_limit,
                bkt_avail, bkt_cap,
                day_used, daily_limit,
                retry_after,
            )
        else:
            logger.debug(
                "Token budget OK  user=%s mode=%s  win=%d/%d  bucket=%.0f  daily=%d/%d",
                user_id[:8], mode,
                win_used, win_limit,
                bkt_avail,
                day_used, daily_limit,
            )

        return result

    except Exception as exc:
        # Fail open — never block users due to Redis downtime
        logger.error("Rate limiter pre-check error (ALLOWING request): %s", exc)
        win_limit, bkt_cap, _, daily_limit = _thresholds(role)
        return TokenRateLimitResult(
            allowed=True,
            window_limit=win_limit,   window_used=0, window_remaining=win_limit,
            bucket_tokens=float(bkt_cap), bucket_capacity=bkt_cap,
            daily_limit=daily_limit,  daily_used=0,  daily_remaining=daily_limit,
        )


def record_token_usage(
    user_id:      str,
    mode:         str,
    actual_tokens: int,
    role:         str = "student",
) -> dict:
    """
    Phase 2 — Record actual token usage (AFTER Gemini responds).

    Call this with the exact token count from response.usage_metadata.
    Writes to all 3 Redis tiers atomically via Lua script.

    Returns updated state dict for logging/debugging.
    Errors are non-fatal: if Redis is down, usage is silently dropped.
    """
    if not RATE_LIMIT_ENABLED or actual_tokens <= 0:
        return {}

    now_us = int(time.time() * 1_000_000)
    sw_key, bkt_key, day_key = _redis_keys(user_id, mode)
    _, bkt_cap, bkt_rate, _ = _thresholds(role)

    try:
        rc = _get_redis()
        _, record_sha = _load_scripts(rc)

        raw = rc.evalsha(
            record_sha,
            3,
            sw_key, bkt_key, day_key,
            str(now_us),          # ARGV[1]
            "60",                 # ARGV[2] window TTL
            str(actual_tokens),   # ARGV[3] actual tokens consumed
            str(bkt_cap),         # ARGV[4]
            str(bkt_rate),        # ARGV[5]
            str(_daily_ttl_secs()), # ARGV[6]
        )

        new_bucket = float(raw[0])
        new_daily  = int(raw[1])

        logger.info(
            "Token usage RECORDED  user=%s mode=%s  tokens=%d  "
            "bucket_remaining=%.0f  daily_total=%d",
            user_id[:8], mode, actual_tokens, new_bucket, new_daily,
        )
        return {
            "tokens_recorded":  actual_tokens,
            "bucket_remaining": int(new_bucket),
            "daily_total":      new_daily,
        }

    except Exception as exc:
        logger.error(
            "Token usage record error (non-fatal): user=%s tokens=%d  %s",
            user_id[:8], actual_tokens, exc,
        )
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Analytics / admin helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_user_usage(user_id: str) -> dict:
    """
    Return current token usage state for a user across all retrieval modes.
    Useful for student dashboard or admin inspection.
    """
    if not RATE_LIMIT_ENABLED:
        return {"enabled": False}

    try:
        rc    = _get_redis()
        now   = time.time()
        today = time.strftime("%Y-%m-%d", time.gmtime())
        cutoff_us = int((now - 60) * 1_000_000)

        result: dict = {
            "enabled":   True,
            "user_id":   user_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "modes":     {},
        }

        for mode in ("ask", "quiz", "summarize"):
            sw_key = f"{_PREFIX}:{user_id}:{mode}:sw"
            entries = rc.zrangebyscore(sw_key, cutoff_us, "+inf")
            win_tokens = 0
            for member in entries:
                colon = member.find(":")
                if colon != -1:
                    try:
                        win_tokens += int(member[colon + 1:])
                    except ValueError:
                        pass
            result["modes"][mode] = {"tokens_last_60s": win_tokens}

        # Daily total (sum across all modes — same counter)
        day_key = f"{_PREFIX}:{user_id}:daily:{today}"
        result["daily_tokens_used"] = int(rc.get(day_key) or 0)

        # Token bucket
        bkt_key  = f"{_PREFIX}:{user_id}:bucket"
        bkt_data = rc.hgetall(bkt_key)
        result["token_bucket"] = {
            "current_tokens": round(float(bkt_data.get("tokens", 0)), 0),
            "last_refill":    bkt_data.get("last_refill", "unknown"),
        }

        return result

    except Exception as exc:
        logger.warning("get_user_usage error: %s", exc)
        return {"enabled": True, "error": str(exc)}


def reset_user_limits(user_id: str) -> int:
    """Admin action: delete all rate-limit state for a user. Returns keys deleted."""
    if not RATE_LIMIT_ENABLED:
        return 0
    try:
        rc   = _get_redis()
        keys = rc.keys(f"{_PREFIX}:{user_id}:*")
        if keys:
            rc.delete(*keys)
        logger.info("Rate limit RESET  user=%s  keys_deleted=%d", user_id[:8], len(keys))
        return len(keys)
    except Exception as exc:
        logger.warning("reset_user_limits error: %s", exc)
        return 0
