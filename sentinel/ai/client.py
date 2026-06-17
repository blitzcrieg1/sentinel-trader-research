"""
Sentinel Trader — AI API Client (aiohttp wrapper).

Thin async wrapper around Gemini or NVIDIA NIM. Handles retries, timeouts,
daily call budgets, and malformed-response tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from datetime import UTC, datetime

import aiohttp
from aiohttp.resolver import AsyncResolver

from sentinel.ai.contract import (
    AiDecision,
    parse_ai_response,
    DECISION_JSON_SCHEMA,
)
from sentinel.ai.prompts import build_system_prompt, build_user_message, append_nvidia_schema_instructions
from sentinel.config import Settings, get_settings
from sentinel.net.dns import GOOGLE_DNS_SERVERS

logger = logging.getLogger(__name__)

class ApiError(Exception):
    def __init__(self, status: int, text: str):
        self.status = status
        self.text = text
        super().__init__(f"HTTP {status}: {text}")

# Transient error types that warrant a retry
_RETRYABLE_EXCEPTIONS = (
    aiohttp.ClientError,
    asyncio.TimeoutError,
)

_MAX_RETRIES: int = 4
_RETRY_AFTER_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def _api_quota_kind(exc: ApiError) -> str | None:
    """Classify a Gemini 429 as per-minute (``rpm``) or per-day (``rpd``) quota."""
    if exc.status != 429:
        return None
    try:
        payload = json.loads(exc.text)
        for detail in payload.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("QuotaFailure"):
                for violation in detail.get("violations", []):
                    quota_id = violation.get("quotaId", "")
                    metric = violation.get("quotaMetric", "")
                    if "PerDay" in quota_id or "PerDay" in metric:
                        return "rpd"
                    if "PerMinute" in quota_id or "PerMinute" in metric:
                        return "rpm"
        message = payload.get("error", {}).get("message", exc.text).lower()
        if "perday" in message or "per day" in message:
            return "rpd"
        if "perminute" in message or "per minute" in message:
            return "rpm"
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None


def _api_retry_after_sec(exc: ApiError) -> float | None:
    """Parse RetryInfo / message from a Gemini 429 response."""
    try:
        payload = json.loads(exc.text)
        for detail in payload.get("error", {}).get("details", []):
            if "RetryInfo" in detail.get("@type", ""):
                delay = detail.get("retryDelay", "")
                if isinstance(delay, str) and delay.endswith("s"):
                    return float(delay[:-1])
        message = payload.get("error", {}).get("message", exc.text)
        match = _RETRY_AFTER_RE.search(message)
        if match:
            return float(match.group(1))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


class _AiRateLimiter:
    """Rolling 60s window — serialises acquire() so bursts queue instead of 429."""

    def __init__(self, max_rpm: int) -> None:
        self._max_rpm = max_rpm
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    def at_capacity(self) -> bool:
        now = time.monotonic()
        while self._timestamps and self._timestamps[0] <= now - 60.0:
            self._timestamps.popleft()
        return len(self._timestamps) >= self._max_rpm

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._timestamps and self._timestamps[0] <= now - 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_rpm:
                    self._timestamps.append(now)
                    return
                wait = 60.0 - (now - self._timestamps[0]) + 0.1
                logger.info(
                    "AI rate limit: %d/%d RPM — waiting %.1fs",
                    len(self._timestamps), self._max_rpm, wait,
                )
                await asyncio.sleep(wait)


_gemini_limiters: dict[str, _AiRateLimiter] = {}
_gemini_limiter_signature: tuple[int, int, int] | None = None
_nvidia_limiter: _AiRateLimiter | None = None
_nvidia_limiter_rpm: int | None = None
_gemini_key_backoff_until: dict[int, float] = {}


class _GeminiKeyRouter:
    """Per-key RPM limiters and transient backoff for gemini_dual."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def rpm_for_slot(self, slot: int) -> int:
        if slot == 1:
            cap = self._settings.gemini_key1_max_rpm
            return cap if cap > 0 else self._settings.ai_max_rpm
        return self._settings.gemini_key2_max_rpm

    def limiter(self, slot: int) -> _AiRateLimiter:
        global _gemini_limiters, _gemini_limiter_signature
        sig = (
            self.rpm_for_slot(1),
            self.rpm_for_slot(2),
            self._settings.ai_max_rpm,
        )
        if _gemini_limiter_signature != sig:
            _gemini_limiters = {}
            _gemini_limiter_signature = sig
        key = f"slot{slot}"
        if key not in _gemini_limiters:
            _gemini_limiters[key] = _AiRateLimiter(self.rpm_for_slot(slot))
        return _gemini_limiters[key]

    def is_in_backoff(self, slot: int) -> bool:
        return time.monotonic() < _gemini_key_backoff_until.get(slot, 0.0)

    def set_backoff(self, slot: int, seconds: float) -> None:
        global _gemini_key_backoff_until
        until = time.monotonic() + max(seconds, 0.0)
        current = _gemini_key_backoff_until.get(slot, 0.0)
        _gemini_key_backoff_until[slot] = max(current, until)

    def try_order(self, key1_ok: bool, key2_ok: bool) -> list[int]:
        """Paid key 1 first; key 2 only when key 1 is unavailable or struggling."""
        order: list[int] = []
        if key1_ok:
            order.append(1)
        if key2_ok:
            order.append(2)
        return order


def _gemini_key_router(settings: Settings) -> _GeminiKeyRouter:
    return _GeminiKeyRouter(settings)


def _nvidia_rate_limiter(settings: Settings) -> _AiRateLimiter:
    global _nvidia_limiter, _nvidia_limiter_rpm
    if _nvidia_limiter is None or _nvidia_limiter_rpm != settings.nvidia_max_rpm:
        _nvidia_limiter = _AiRateLimiter(settings.nvidia_max_rpm)
        _nvidia_limiter_rpm = settings.nvidia_max_rpm
    return _nvidia_limiter

_cerebras_limiter: _AiRateLimiter | None = None
_cerebras_limiter_rpm: int | None = None

def _cerebras_rate_limiter(settings: Settings) -> _AiRateLimiter:
    global _cerebras_limiter, _cerebras_limiter_rpm
    if _cerebras_limiter is None or _cerebras_limiter_rpm != settings.cerebras_max_rpm:
        _cerebras_limiter = _AiRateLimiter(settings.cerebras_max_rpm)
        _cerebras_limiter_rpm = settings.cerebras_max_rpm
    return _cerebras_limiter


class AiClient:
    """Async wrapper for Gemini, gemini_dual, NVIDIA NIM, or Gemini+NVIDIA dual."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the AI client."""
        self._settings: Settings = settings or get_settings()
        self._provider = self._settings.ai_provider or "gemini"
        self._cerebras_exhausted: bool = False  # set True on 402; switches to gemini_dual
        self._nvidia_fallbacks: int = 0
        self._gemini_secondary_calls: int = 0
        self._gemini_key1_calls: int = 0
        self._gemini_key2_calls: int = 0

        self._timeout = aiohttp.ClientTimeout(total=float(self._settings.ai_timeout_sec))
        resolver = AsyncResolver(nameservers=list(GOOGLE_DNS_SERVERS))
        connector = aiohttp.TCPConnector(resolver=resolver, family=0, limit=10)
        self._gemini_session = aiohttp.ClientSession(
            connector=connector, timeout=self._timeout,
        )
        self._nvidia_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10), timeout=self._timeout,
        )
        self._cerebras_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10), timeout=self._timeout,
        )

        # Budget tracking (resets at midnight UTC via reset_daily_budget)
        self._calls_today: int = 0
        self._budget_date: str = datetime.now(UTC).strftime("%Y-%m-%d")

        # Malformed-response tracking
        self._consecutive_malformed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close persistent HTTP sessions. Safe to call multiple times."""
        for session in (self._gemini_session, self._nvidia_session, self._cerebras_session):
            if not session.closed:
                await session.close()

    async def request_decision(
        self,
        feature_dict: dict,
        active_positions: list,
        recent_decisions: list,
        past_lessons: list[str] | None = None,
        recent_news: list[str] | None = None,
        recent_vetoes: list[dict] | None = None,
    ) -> tuple[AiDecision | None, dict]:
        meta: dict = {
            "model": self._settings.ai_model,
            "provider": self._provider,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "error": None,
        }

        # -- Budget guard --------------------------------------------------
        self._maybe_rotate_budget_date()
        if self._calls_today >= self._settings.max_ai_calls_per_day:
            logger.warning(
                "AI daily budget exhausted (%d/%d). Returning no decision.",
                self._calls_today,
                self._settings.max_ai_calls_per_day,
            )
            meta["error"] = "daily_budget_exhausted"
            return None, meta

        # -- Build request -------------------------------------------------
        # System prompt is regime-aware: strategy directives change with
        # the market regime computed by the feature engine.
        system_prompt = build_system_prompt(feature_dict.get("market_regime"))
        user_content = build_user_message(
            feature_dict=feature_dict,
            active_positions=active_positions,
            recent_decisions=recent_decisions,
            past_lessons=past_lessons,
            recent_news=recent_news,
            recent_vetoes=recent_vetoes,
        )

        # -- API call with retries -----------------------------------------
        raw_text: str | None = None
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                raw_text, meta = await self._call_api(system_prompt, user_content, meta)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                is_retryable = isinstance(exc, _RETRYABLE_EXCEPTIONS)
                if isinstance(exc, ApiError) and exc.status in (429, 500, 502, 503, 504):
                    is_retryable = True
                if isinstance(exc, ApiError) and exc.status == 429:
                    if _api_quota_kind(exc) == "rpd" or self.gemini_daily_exhausted():
                        is_retryable = False
                    elif exc.text == "both Gemini keys hit daily call cap":
                        is_retryable = False

                if not is_retryable:
                    logger.error("AI call failed with non-retryable error: %s", exc)
                    meta["error"] = f"{type(exc).__name__}: {exc}"
                    self._calls_today += 1
                    return None, meta

                wait = _api_retry_after_sec(exc) if isinstance(exc, ApiError) else None
                if wait is None:
                    wait = float(2 ** attempt)
                else:
                    wait = min(wait + 1.0, 120.0)
                logger.warning(
                    "AI call attempt %d/%d failed (%s: %s). Retrying in %.0fs.",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    type(exc).__name__,
                    exc,
                    wait,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(wait)

        if last_error is not None:
            logger.error(
                "AI call failed after %d attempts: %s",
                _MAX_RETRIES + 1,
                last_error,
            )
            meta["error"] = f"{type(last_error).__name__}: {last_error}"
            self._calls_today += 1
            return None, meta

        # -- Increment call counter ----------------------------------------
        self._calls_today += 1

        # -- Parse response ------------------------------------------------
        if not raw_text:
            logger.error("AI returned empty response text")
            meta["error"] = "empty_response"
            self._consecutive_malformed += 1
            return None, meta

        decision = parse_ai_response(raw_text)

        if decision is None:
            self._consecutive_malformed += 1
            logger.error(
                "Malformed AI response (%d consecutive). Raw: %.500s",
                self._consecutive_malformed,
                raw_text,
            )
            meta["error"] = "malformed_response"
            meta["raw_text"] = raw_text[:2000]
            return None, meta

        # Successful parse → reset malformed counter
        self._consecutive_malformed = 0

        logger.info(
            "AI decision: symbol=%s decision=%s confidence=%.2f "
            "provider=%s input_tokens=%d output_tokens=%d latency=%dms",
            decision.symbol,
            decision.decision,
            decision.confidence,
            meta.get("provider", self._provider),
            meta["input_tokens"],
            meta["output_tokens"],
            meta["latency_ms"],
        )

        return decision, meta

    def get_budget_status(self) -> dict:
        self._maybe_rotate_budget_date()
        return {
            "calls_today": self._calls_today,
            "calls_remaining": max(
                0, self._settings.max_ai_calls_per_day - self._calls_today
            ),
            "consecutive_malformed": self._consecutive_malformed,
            "budget_date": self._budget_date,
            "nvidia_fallbacks_today": self._nvidia_fallbacks,
            "gemini_secondary_calls_today": self._gemini_secondary_calls,
            "gemini_key1_calls_today": self._gemini_key1_calls,
            "gemini_key2_calls_today": self._gemini_key2_calls,
            "gemini_daily_exhausted": self.gemini_daily_exhausted(),
        }

    def reset_daily_budget(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        logger.info(
            "Resetting AI daily budget. Previous: %d calls on %s.",
            self._calls_today,
            self._budget_date,
        )
        self._calls_today = 0
        self._gemini_key1_calls = 0
        self._gemini_key2_calls = 0
        self._gemini_secondary_calls = 0
        self._nvidia_fallbacks = 0
        self._budget_date = today

    @property
    def consecutive_malformed(self) -> int:
        return self._consecutive_malformed

    def _gemini_key1_daily_exhausted(self) -> bool:
        cap = self._settings.gemini_key1_max_daily_calls
        return cap > 0 and self._gemini_key1_calls >= cap

    def _gemini_key2_daily_exhausted(self) -> bool:
        cap = self._settings.gemini_key2_max_daily_calls
        return cap > 0 and self._gemini_key2_calls >= cap

    def gemini_daily_exhausted(self) -> bool:
        """True when both Gemini keys have hit their daily routing caps."""
        if self._provider != "gemini_dual":
            return False
        return self._gemini_key1_daily_exhausted() and self._gemini_key2_daily_exhausted()

    def _force_gemini_key_daily_exhausted(self, slot: int) -> None:
        cap = (
            self._settings.gemini_key1_max_daily_calls
            if slot == 1
            else self._settings.gemini_key2_max_daily_calls
        )
        if cap <= 0:
            cap = 20
        if slot == 1:
            self._gemini_key1_calls = max(self._gemini_key1_calls, cap)
        else:
            self._gemini_key2_calls = max(self._gemini_key2_calls, cap)

    def _note_gemini_quota_violation(self, exc: ApiError, slot: int) -> None:
        kind = _api_quota_kind(exc)
        if kind == "rpd":
            self._force_gemini_key_daily_exhausted(slot)
            logger.warning(
                "Gemini key %d hit Google daily quota — marking exhausted for today", slot,
            )

    def _record_gemini_key_call(self, slot: int) -> None:
        if slot == 1:
            self._gemini_key1_calls += 1
        else:
            self._gemini_key2_calls += 1
            self._gemini_secondary_calls += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _call_api(
        self, system_prompt: str, user_content: str, meta: dict,
    ) -> tuple[str, dict]:
        if self._provider == "cerebras" and not self._cerebras_exhausted:
            await _cerebras_rate_limiter(self._settings).acquire()
            try:
                return await self._call_cerebras(system_prompt, user_content, meta)
            except ApiError as exc:
                if exc.status == 402:
                    self._cerebras_exhausted = True
                    logger.error(
                        "CEREBRAS BALANCE EXHAUSTED (402) — auto-switching to gemini_dual. "
                        "Top up at cloud.cerebras.ai or set CEREBRAS_MODEL=llama3.3-70b (free)."
                    )
                    # fall through to gemini_dual below
                else:
                    raise

        if self._provider == "cerebras" and self._cerebras_exhausted:
            return await self._call_gemini_dual(system_prompt, user_content, meta)

        if self._provider == "nvidia":
            await _nvidia_rate_limiter(self._settings).acquire()
            return await self._call_nvidia(system_prompt, user_content, meta)

        if self._provider == "gemini_dual":
            return await self._call_gemini_dual(system_prompt, user_content, meta)

        if self._provider == "dual" and _gemini_key_router(self._settings).limiter(1).at_capacity():
            logger.info("Gemini RPM full — routing to NVIDIA NIM")
            return await self._nvidia_fallback(system_prompt, user_content, meta)

        await _gemini_key_router(self._settings).limiter(1).acquire()
        try:
            return await self._call_gemini(
                system_prompt, user_content, meta,
                api_key=self._settings.gemini_api_key,
            )
        except ApiError as exc:
            if (
                self._settings.nvidia_fallback_enabled
                and exc.status in (429, 500, 502, 503, 504)
            ):
                logger.warning(
                    "Gemini HTTP %d — falling back to NVIDIA NIM", exc.status,
                )
                return await self._nvidia_fallback(system_prompt, user_content, meta)
            raise

    async def _call_gemini_dual(
        self, system_prompt: str, user_content: str, meta: dict,
    ) -> tuple[str, dict]:
        """Dual Gemini: paid key 1 primary; key 2 only on error/backoff."""
        key1_ok = not self._gemini_key1_daily_exhausted()
        key2_ok = not self._gemini_key2_daily_exhausted()
        if not key1_ok and not key2_ok:
            raise ApiError(429, "both Gemini keys hit daily call cap")

        router = _gemini_key_router(self._settings)
        last_exc: ApiError | None = None

        for slot in router.try_order(key1_ok, key2_ok):
            if router.is_in_backoff(slot):
                logger.info(
                    "Gemini key %d in backoff — skipping to next key", slot,
                )
                continue

            api_key = (
                self._settings.gemini_api_key
                if slot == 1
                else self._settings.gemini_api_key_2
            )
            await router.limiter(slot).acquire()
            try:
                result = await self._call_gemini(
                    system_prompt, user_content, meta,
                    api_key=api_key,
                    provider_label=f"gemini_{slot}",
                )
            except ApiError as exc:
                last_exc = exc
                self._note_gemini_quota_violation(exc, slot)
                if exc.status in (429, 500, 502, 503, 504):
                    wait = _api_retry_after_sec(exc) or 5.0
                    router.set_backoff(slot, min(wait + 1.0, 120.0))
                    logger.warning(
                        "Gemini key %d HTTP %d — backing off %.0fs, trying other key",
                        slot, exc.status, wait,
                    )
                    continue
                raise

            self._record_gemini_key_call(slot)
            out_meta = result[1]
            if slot == 2:
                out_meta = {**out_meta, "fallback": True}
            return result[0], out_meta

        if last_exc is not None:
            raise last_exc
        raise ApiError(429, "both Gemini keys temporarily unavailable")

    async def _gemini_dual_text(self, prompt: str, temperature: float) -> str:
        key1_ok = not self._gemini_key1_daily_exhausted()
        key2_ok = not self._gemini_key2_daily_exhausted()
        router = _gemini_key_router(self._settings)
        last_exc: ApiError | None = None

        for slot in router.try_order(key1_ok, key2_ok):
            if router.is_in_backoff(slot):
                continue
            api_key = (
                self._settings.gemini_api_key
                if slot == 1
                else self._settings.gemini_api_key_2
            )
            await router.limiter(slot).acquire()
            try:
                text = await self._gemini_text(prompt, temperature, api_key=api_key)
            except ApiError as exc:
                last_exc = exc
                self._note_gemini_quota_violation(exc, slot)
                if exc.status in (429, 500, 502, 503, 504):
                    wait = _api_retry_after_sec(exc) or 5.0
                    router.set_backoff(slot, min(wait + 1.0, 120.0))
                    continue
                raise
            self._record_gemini_key_call(slot)
            return text

        if last_exc is not None:
            raise last_exc
        raise ApiError(429, "both Gemini keys temporarily unavailable")

    async def _nvidia_fallback(
        self, system_prompt: str, user_content: str, meta: dict,
    ) -> tuple[str, dict]:
        await _nvidia_rate_limiter(self._settings).acquire()
        self._nvidia_fallbacks += 1
        result = await self._call_nvidia(system_prompt, user_content, meta)
        meta = {**result[1], "provider": "nvidia", "fallback": True}
        return result[0], meta

    async def request_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Plain-text completion (post-mortem reflection). Never raises."""
        try:
            if self._provider == "cerebras" and not self._cerebras_exhausted:
                await _cerebras_rate_limiter(self._settings).acquire()
                try:
                    return await self._cerebras_text(prompt, temperature)
                except ApiError as exc:
                    if exc.status == 402:
                        self._cerebras_exhausted = True
                        logger.warning("Cerebras balance exhausted (402) during reflection. Switching to gemini_dual.")
                    else:
                        raise

            if self._provider == "cerebras" and self._cerebras_exhausted:
                return await self._gemini_dual_text(prompt, temperature)

            if self._provider == "nvidia":
                await _nvidia_rate_limiter(self._settings).acquire()
                return await self._nvidia_text(prompt, temperature)

            if self._provider == "gemini_dual":
                return await self._gemini_dual_text(prompt, temperature)

            if self._provider == "dual" and _gemini_key_router(self._settings).limiter(1).at_capacity():
                await _nvidia_rate_limiter(self._settings).acquire()
                return await self._nvidia_text(prompt, temperature)

            await _gemini_key_router(self._settings).limiter(1).acquire()
            try:
                return await self._gemini_text(
                    prompt, temperature, api_key=self._settings.gemini_api_key,
                )
            except ApiError as exc:
                if self._settings.nvidia_fallback_enabled and exc.status == 429:
                    await _nvidia_rate_limiter(self._settings).acquire()
                    return await self._nvidia_text(prompt, temperature)
                raise
        except Exception as exc:  # noqa: BLE001
            logger.error("request_text failed: %s", exc)
            return None

    async def _call_gemini(
        self,
        system_prompt: str,
        user_content: str,
        meta: dict,
        *,
        api_key: str | None = None,
        provider_label: str = "gemini",
    ) -> tuple[str, dict]:
        start = time.monotonic()
        model = self._settings.gemini_model
        api_key = api_key or self._settings.gemini_api_key
        if model.startswith("projects/"):
            region = model.split("/")[3]
            url = f"https://{region}-aiplatform.googleapis.com/v1/{model}:generateContent"
        elif model.startswith("tunedModels/"):
            url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": DECISION_JSON_SCHEMA,
            },
        }

        async with self._gemini_session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ApiError(resp.status, text)
            data = await resp.json()

        elapsed_ms = int((time.monotonic() - start) * 1000)
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

        raw_text = ""
        stop_reason = None
        candidates = data.get("candidates", [])
        if candidates:
            stop_reason = candidates[0].get("finishReason")
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                raw_text = parts[0].get("text", "")

        meta = {
            **meta,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": elapsed_ms,
            "model": model,
            "provider": provider_label,
            "stop_reason": stop_reason,
        }
        return raw_text, meta

    async def _call_nvidia(
        self, system_prompt: str, user_content: str, meta: dict,
    ) -> tuple[str, dict]:
        start = time.monotonic()
        model = self._settings.nvidia_model
        api_key = self._settings.nvidia_api_key
        nvidia_system = append_nvidia_schema_instructions(system_prompt)

        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": nvidia_system},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens": 1024,
            "stream": False
        }

        async with self._nvidia_session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ApiError(resp.status, text)
            data = await resp.json()

        elapsed_ms = int((time.monotonic() - start) * 1000)

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        choices = data.get("choices", [])
        raw_text = ""
        stop_reason = None
        if choices:
            choice = choices[0]
            stop_reason = choice.get("finish_reason")
            raw_text = choice.get("message", {}).get("content", "")

        meta = {
            **meta,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": elapsed_ms,
            "model": model,
            "provider": "nvidia",
            "stop_reason": stop_reason,
        }

        logger.debug(
            "API response: tokens=%d/%d latency=%dms stop=%s",
            input_tokens,
            output_tokens,
            elapsed_ms,
            meta.get("stop_reason"),
        )

        return raw_text, meta

    async def _gemini_text(
        self, prompt: str, temperature: float, *, api_key: str | None = None,
    ) -> str:
        model = self._settings.gemini_model
        api_key = api_key or self._settings.gemini_api_key
        if model.startswith("projects/"):
            region = model.split("/")[3]
            url = f"https://{region}-aiplatform.googleapis.com/v1/{model}:generateContent"
        elif model.startswith("tunedModels/"):
            url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": api_key}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        async with self._gemini_session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ApiError(resp.status, text)
            data = await resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def _nvidia_text(self, prompt: str, temperature: float) -> str:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._settings.nvidia_api_key}",
        }
        payload = {
            "model": self._settings.nvidia_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 512,
            "stream": False,
        }
        async with self._nvidia_session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ApiError(resp.status, text)
            data = await resp.json()
        return data["choices"][0]["message"]["content"].strip()

    async def _call_cerebras(
        self, system_prompt: str, user_content: str, meta: dict,
    ) -> tuple[str, dict]:
        start = time.monotonic()
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._settings.cerebras_api_key}",
        }
        payload = {
            "model": self._settings.cerebras_model,
            "messages": [
                {"role": "system", "content": append_nvidia_schema_instructions(system_prompt)},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
            "stream": False,
        }
        async with self._cerebras_session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ApiError(resp.status, text)
            data = await resp.json()

        elapsed_ms = int((time.monotonic() - start) * 1000)
        usage = data.get("usage", {})
        choices = data.get("choices", [])
        raw_text = ""
        stop_reason = None
        if choices:
            stop_reason = choices[0].get("finish_reason")
            raw_text = choices[0].get("message", {}).get("content", "")

        meta = {
            **meta,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "latency_ms": elapsed_ms,
            "model": self._settings.cerebras_model,
            "provider": "cerebras",
            "stop_reason": stop_reason,
        }
        return raw_text, meta

    async def _cerebras_text(self, prompt: str, temperature: float) -> str:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._settings.cerebras_api_key}",
        }
        payload = {
            "model": self._settings.cerebras_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 512,
            "stream": False,
        }
        async with self._cerebras_session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ApiError(resp.status, text)
            data = await resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        if not content:
            raise ValueError(f"Cerebras response missing content: {data}")
        return content.strip()

    def _maybe_rotate_budget_date(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._budget_date:
            logger.info(
                "UTC date rolled over (%s -> %s). Auto-resetting AI budget.",
                self._budget_date,
                today,
            )
            self._calls_today = 0
            self._gemini_key1_calls = 0
            self._gemini_key2_calls = 0
            self._gemini_secondary_calls = 0
            self._nvidia_fallbacks = 0
            self._budget_date = today
