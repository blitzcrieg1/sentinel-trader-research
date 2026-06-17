"""
Sentinel Trader — Silent-Failure Watchdog.

Background task that detects the bot going *quietly* dead — the failure
mode where the process is up, systemd reports ``active``, but no work is
actually landing in the database (e.g. the SQLite file lost write
permissions, the scheduler stalled, or every scan is erroring out).

It works by reading the newest ``pipeline_runs.created_at`` timestamp. If
that timestamp stops advancing for longer than ``stale_threshold``, the
watchdog fires a single Telegram alert and re-arms only once writes resume,
so a sustained outage produces one alert, not a storm.

The watchdog is strictly *read-only*: it never writes to the database, so
it keeps working even during the exact readonly-DB incident it exists to
catch.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from sentinel.core.pipeline import PipelineContext
from sentinel.store import get_connection

logger = logging.getLogger(__name__)

#: How often to poll the DB for liveness, in seconds.
_CHECK_INTERVAL_SEC = 300

#: Minimum staleness floor (seconds). The effective threshold is the larger
#: of this and 2× the scan interval, so slow scan cadences aren't false-flagged.
_MIN_STALE_THRESHOLD_SEC = 1800


async def run_watchdog(ctx: PipelineContext, stop_event: asyncio.Event) -> None:
    """Alert when no pipeline run has been persisted for too long.

    Runs until ``stop_event`` is set. Every failure inside the loop is
    swallowed and retried — the watchdog must never be the thing that
    crashes the supervisor.
    """
    scan_interval_sec = ctx.cfg.scan_interval_minutes * 60
    stale_threshold = max(_MIN_STALE_THRESHOLD_SEC, scan_interval_sec * 2)
    logger.info(
        "watchdog started (poll=%ds, stale_threshold=%ds)",
        _CHECK_INTERVAL_SEC, stale_threshold,
    )

    alerted = False  # True while an outage is active — suppresses repeat alerts

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_CHECK_INTERVAL_SEC)
            break  # stop_event set
        except TimeoutError:
            pass  # poll interval elapsed

        try:
            age_sec = await _latest_pipeline_age_sec()
        except Exception as exc:  # noqa: BLE001 — never crash the supervisor
            logger.exception("watchdog liveness probe failed: %s", exc)
            continue

        if age_sec is None:
            continue  # no runs yet (fresh DB) — nothing to compare against

        if age_sec > stale_threshold and not alerted:
            alerted = True
            mins = age_sec / 60.0
            logger.error(
                "WATCHDOG: no pipeline run persisted for %.0f min "
                "(threshold %.0f min) — bot may be silently stuck",
                mins, stale_threshold / 60.0,
            )
            await ctx.notify(
                f"⚠️ WATCHDOG ALERT\n"
                f"No pipeline activity for {mins:.0f} min "
                f"(threshold {stale_threshold / 60.0:.0f} min).\n"
                f"The bot is running but not writing — check DB permissions, "
                f"the scheduler, and `journalctl -u sentinel-trader`."
            )
        elif age_sec <= stale_threshold and alerted:
            alerted = False
            logger.info("WATCHDOG: pipeline activity resumed (age %.0fs)", age_sec)
            await ctx.notify("✅ WATCHDOG: pipeline activity resumed — bot is writing again.")

    logger.info("watchdog stopped")


async def _latest_pipeline_age_sec() -> float | None:
    """Seconds since the most recent pipeline run was persisted, or None."""
    async with get_connection() as db:
        cursor = await db.execute("SELECT MAX(created_at) FROM pipeline_runs")
        row = await cursor.fetchone()

    if not row or row[0] is None:
        return None

    latest = _parse_utc(str(row[0]))
    if latest is None:
        return None
    return (datetime.now(UTC) - latest).total_seconds()


def _parse_utc(ts: str) -> datetime | None:
    """Parse the bot's ``%Y-%m-%dT%H:%M:%SZ`` timestamps into aware datetimes."""
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
