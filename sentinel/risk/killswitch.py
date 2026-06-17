"""
Sentinel Trader — Kill Switch.

A persisted halt flag in SQLite that survives process restarts, plus the
auto-trigger counters (consecutive execution errors, malformed AI responses,
consecutive losses) that escalate into a halt.

The kill switch itself only manages **state and events** — it never talks
to the broker. The emergency sequence (cancel orders → close positions) is
performed by the caller (pipeline / admin layer) via the optional
``emergency_callback``, so this module stays broker-agnostic and trivially
testable.

All state lives in the ``state`` key/value table so a restarted process
resumes exactly where it halted (spec §8: kill switch is persisted).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import aiosqlite

from sentinel.config import Settings, get_settings
from sentinel.store.models import Event
from sentinel.store.repo import get_state, insert_event, set_state

logger = logging.getLogger(__name__)

# ── Persisted state keys ─────────────────────────────────────────────────
KEY_HALTED: Final[str] = "killswitch.halted"
KEY_HALT_REASON: Final[str] = "killswitch.halt_reason"
KEY_HALTED_AT: Final[str] = "killswitch.halted_at"
KEY_HALT_EXPIRES_AT: Final[str] = "killswitch.halt_expires_at"  # empty = manual /resume only
KEY_CONSEC_EXEC_ERRORS: Final[str] = "killswitch.consecutive_exec_errors"
KEY_CONSEC_MALFORMED: Final[str] = "killswitch.consecutive_malformed"
KEY_CONSEC_LOSSES: Final[str] = "killswitch.consecutive_losses"
KEY_LAST_LOSS_AT: Final[str] = "killswitch.last_loss_at"

#: Async callback invoked on engage: must cancel all orders + close all positions.
EmergencyCallback = Callable[[str], Awaitable[None]]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        logger.error("corrupt timestamp in state table: %r", raw)
        return None


@dataclass(frozen=True, slots=True)
class HaltStatus:
    """Current kill-switch state, read from SQLite."""

    halted: bool
    reason: str | None
    halted_at: datetime | None
    expires_at: datetime | None  # None = requires manual /resume


class KillSwitch:
    """Persisted halt flag + auto-trigger counters.

    All methods take an open ``aiosqlite.Connection`` so the caller controls
    transaction scope and connection lifetime.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        emergency_callback: EmergencyCallback | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._emergency_callback = emergency_callback

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    async def status(self, db: aiosqlite.Connection) -> HaltStatus:
        """Read the full halt status. Auto-clears an expired timed halt."""
        halted_raw = await get_state(db, KEY_HALTED)
        halted = halted_raw == "true"

        if not halted:
            return HaltStatus(halted=False, reason=None, halted_at=None, expires_at=None)

        expires_at = _parse_iso(await get_state(db, KEY_HALT_EXPIRES_AT))
        if expires_at is not None and _utcnow() >= expires_at:
            await self._clear(db, "timed halt expired")
            return HaltStatus(halted=False, reason=None, halted_at=None, expires_at=None)

        return HaltStatus(
            halted=True,
            reason=await get_state(db, KEY_HALT_REASON),
            halted_at=_parse_iso(await get_state(db, KEY_HALTED_AT)),
            expires_at=expires_at,
        )

    async def is_halted(self, db: aiosqlite.Connection) -> bool:
        """Convenience: True if trading is currently halted."""
        return (await self.status(db)).halted

    # ------------------------------------------------------------------
    # Engage / release
    # ------------------------------------------------------------------

    async def engage(
        self,
        db: aiosqlite.Connection,
        reason: str,
        *,
        context: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Engage the kill switch.

        Sequence (spec §8): persist halted flag → emergency callback (cancel
        orders, close positions) → log critical event. The flag is persisted
        before the flatten so that a failure (or crash) while flattening can
        never leave the system un-halted.

        Args:
            reason: Human-readable trigger description.
            context: Extra structured context for the event log.
            expires_at: Optional auto-expiry (used by the 24h daily-loss
                halt). ``None`` = only a manual ``/resume`` releases it.
        """
        if await self.is_halted(db):
            logger.warning("kill switch already engaged; ignoring engage(%s)", reason)
            return

        logger.critical("KILL SWITCH ENGAGED: %s", reason)

        # Persist the halt flag FIRST: a crash during the emergency flatten
        # must never leave the system un-halted on restart.
        now = _utcnow()
        await set_state(db, KEY_HALTED, "true")
        await set_state(db, KEY_HALT_REASON, reason)
        await set_state(db, KEY_HALTED_AT, _iso(now))
        await set_state(db, KEY_HALT_EXPIRES_AT, _iso(expires_at) if expires_at else "")

        callback_error: str | None = None
        if self._emergency_callback is not None:
            try:
                await self._emergency_callback(reason)
            except Exception as exc:
                callback_error = f"{type(exc).__name__}: {exc}"
                logger.critical(
                    "emergency callback FAILED during kill switch engage: %s",
                    callback_error,
                )

        await insert_event(db, Event(
            event_type="kill_switch",
            severity="critical",
            message=f"Kill switch engaged: {reason}",
            context_json=json.dumps({
                "reason": reason,
                "expires_at": _iso(expires_at) if expires_at else None,
                "emergency_callback_error": callback_error,
                **(context or {}),
            }),
        ))

    async def release(self, db: aiosqlite.Connection, released_by: str = "admin") -> None:
        """Release the halt (only via explicit ``/resume``). Resets all counters."""
        await self._clear(db, f"released by {released_by}")
        await self.reset_counters(db)
        await insert_event(db, Event(
            event_type="resume",
            severity="info",
            message=f"Kill switch released by {released_by}",
        ))
        logger.warning("kill switch released by %s", released_by)

    async def _clear(self, db: aiosqlite.Connection, why: str) -> None:
        await set_state(db, KEY_HALTED, "false")
        await set_state(db, KEY_HALT_REASON, "")
        await set_state(db, KEY_HALT_EXPIRES_AT, "")
        logger.info("halt flag cleared (%s)", why)

    # ------------------------------------------------------------------
    # Auto-trigger counters
    # ------------------------------------------------------------------

    async def _bump_counter(self, db: aiosqlite.Connection, key: str) -> int:
        current = int(await get_state(db, key) or 0)
        new_value = current + 1
        await set_state(db, key, str(new_value))
        return new_value

    async def record_execution_error(self, db: aiosqlite.Connection, detail: str) -> None:
        """Record one execution failure; engages at the configured threshold."""
        count = await self._bump_counter(db, KEY_CONSEC_EXEC_ERRORS)
        logger.error("execution error %d/%d: %s",
                     count, self._settings.max_consecutive_exec_errors, detail)
        if count >= self._settings.max_consecutive_exec_errors:
            await self.engage(
                db,
                f"{count} consecutive execution errors",
                context={"last_error": detail},
            )

    async def record_execution_success(self, db: aiosqlite.Connection) -> None:
        await set_state(db, KEY_CONSEC_EXEC_ERRORS, "0")

    async def record_malformed_response(self, db: aiosqlite.Connection, detail: str) -> None:
        """Record one malformed AI response; engages at the configured threshold."""
        count = await self._bump_counter(db, KEY_CONSEC_MALFORMED)
        logger.error("malformed AI response %d/%d: %s",
                     count, self._settings.max_consecutive_malformed, detail)
        if count >= self._settings.max_consecutive_malformed:
            await self.engage(
                db,
                f"{count} consecutive malformed AI responses (model may be degraded)",
                context={"last_error": detail},
            )

    async def record_valid_response(self, db: aiosqlite.Connection) -> None:
        await set_state(db, KEY_CONSEC_MALFORMED, "0")

    async def record_trade_loss(self, db: aiosqlite.Connection, pnl_usdt: float) -> None:
        """Record a losing trade close — feeds cooldown + consecutive-loss gates."""
        count = await self._bump_counter(db, KEY_CONSEC_LOSSES)
        await set_state(db, KEY_LAST_LOSS_AT, _iso(_utcnow()))
        logger.warning("losing trade recorded: pnl=%.4f (consecutive: %d/%d)",
                       pnl_usdt, count, self._settings.max_consecutive_losses)
        # Soft gate only — risk engine blocks new entries; no manual /resume halt.

    async def record_trade_win(self, db: aiosqlite.Connection) -> None:
        await set_state(db, KEY_CONSEC_LOSSES, "0")

    async def reset_counters(self, db: aiosqlite.Connection) -> None:
        """Zero all auto-trigger counters (called on /resume)."""
        for key in (KEY_CONSEC_EXEC_ERRORS, KEY_CONSEC_MALFORMED, KEY_CONSEC_LOSSES):
            await set_state(db, key, "0")

    # ------------------------------------------------------------------
    # Cooldown / counter reads (consumed by the risk engine)
    # ------------------------------------------------------------------

    async def seconds_since_last_loss(self, db: aiosqlite.Connection) -> float | None:
        """Seconds elapsed since the last losing trade closed; None if never."""
        last_loss = _parse_iso(await get_state(db, KEY_LAST_LOSS_AT))
        if last_loss is None:
            return None
        return (_utcnow() - last_loss).total_seconds()

    async def consecutive_losses(self, db: aiosqlite.Connection) -> int:
        return int(await get_state(db, KEY_CONSEC_LOSSES) or 0)
