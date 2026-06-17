"""
Sentinel Trader — Candle-Close-Aligned Scheduler.

Fires the analysis pipeline exactly on 15m / 1h candle boundaries
(00:00, 00:15, 00:30, ... for 15m; top of the hour for 1h), plus a small
configurable jitter so the exchange has definitely printed the final candle
before we fetch it.

Behaviour guarantees:
- **Boundary math is pure and testable** (``next_candle_close``).
- **Missed ticks are skipped, never replayed.** If a cycle (or a long GC /
  network stall) overruns past one or more boundaries, the loop logs the
  miss and aligns to the *next future* boundary — stale candles are worse
  than no candles.
- **Per-symbol pipelines run concurrently** via ``asyncio.create_task``;
  one slow symbol never delays the others. Task references are retained
  until completion (no premature garbage collection).
- **Graceful shutdown** via a shared ``asyncio.Event``; in-flight pipelines
  are awaited before the loop exits.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import UTC, datetime, timedelta

from sentinel.core.pipeline import PipelineContext, run_analysis_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure boundary math
# ---------------------------------------------------------------------------


def next_candle_close(now: datetime, interval_minutes: int) -> datetime:
    """Return the next candle-close boundary strictly after ``now``.

    Boundaries are aligned to midnight UTC, every ``interval_minutes``
    (e.g. 15 → :00/:15/:30/:45; 60 → top of each hour).
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if interval_minutes <= 0 or (24 * 60) % interval_minutes != 0:
        raise ValueError(f"interval_minutes must evenly divide a day, got {interval_minutes}")

    now_utc = now.astimezone(UTC)
    midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    interval_sec = interval_minutes * 60
    seconds_into_day = (now_utc - midnight).total_seconds()
    boundaries_passed = int(seconds_into_day // interval_sec)
    return midnight + timedelta(seconds=(boundaries_passed + 1) * interval_sec)


def timeframe_label(interval_minutes: int) -> str:
    """Map a minute interval to a CCXT timeframe label ('15m', '1h', '4h')."""
    if interval_minutes % 60 == 0:
        return f"{interval_minutes // 60}h"
    return f"{interval_minutes}m"


async def run_staggered_pipeline(
    ctx: PipelineContext,
    symbol: str,
    timeframe: str,
    *,
    delay: float,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Sleep then run one symbol pipeline — shared by scheduler and /scan."""
    if delay > 0:
        if stop_event is not None:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
                return
            except TimeoutError:
                pass
        else:
            await asyncio.sleep(delay)
    if stop_event is not None and stop_event.is_set():
        return
    await run_analysis_pipeline(ctx, symbol, timeframe)


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------


async def run_scan_scheduler(ctx: PipelineContext, stop_event: asyncio.Event) -> None:
    """Main scan loop: sleep to each candle close (+jitter), fire all symbols.

    Runs until ``stop_event`` is set. Never raises — any unexpected error
    is logged and the loop realigns to the next boundary (systemd-style
    resilience without a process restart).
    """
    cfg = ctx.cfg
    interval = cfg.scan_interval_minutes
    timeframe = timeframe_label(interval)
    in_flight: set[asyncio.Task[None]] = set()

    logger.info(
        "scheduler started: interval=%dmin timeframe=%s symbols=%s jitter=[%.1f, %.1f]s",
        interval, timeframe, cfg.scan_symbols, cfg.scan_jitter_min_sec, cfg.scan_jitter_max_sec,
    )

    last_fired_boundary: datetime | None = None
    is_first_run: bool = True

    while not stop_event.is_set():
        try:
            now = datetime.now(UTC)
            boundary = next_candle_close(now, interval)

            # Missed-tick detection: if we overran past boundaries since the
            # last firing, report how many were skipped.
            if last_fired_boundary is not None:
                missed = int(
                    (boundary - last_fired_boundary).total_seconds() // (interval * 60)
                ) - 1
                if missed > 0:
                    logger.warning(
                        "scheduler missed %d boundary(ies) between %s and %s — skipping, "
                        "aligning to next close",
                        missed, last_fired_boundary.isoformat(), boundary.isoformat(),
                    )

            if is_first_run:
                wait_sec = 0.0
                logger.info("scheduler: firing immediately for the first run")
            else:
                jitter = random.uniform(cfg.scan_jitter_min_sec, cfg.scan_jitter_max_sec)
                wait_sec = (boundary - now).total_seconds() + jitter
                logger.debug(
                    "sleeping %.1fs until %s (+%.1fs jitter)",
                    wait_sec, boundary.isoformat(), jitter,
                )

            is_first_run = False

            # Sleep, but wake immediately if shutdown is requested.
            if wait_sec > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=wait_sec)
                    break  # stop_event set during the sleep
                except TimeoutError:
                    pass  # normal path: boundary reached

            last_fired_boundary = boundary

            stagger = cfg.scan_stagger_sec
            for i, symbol in enumerate(cfg.scan_symbols):
                if stop_event.is_set():
                    logger.info("scheduler: shutdown requested — skipping remaining symbols")
                    break
                task = asyncio.create_task(
                    run_staggered_pipeline(
                        ctx, symbol, timeframe,
                        delay=i * stagger,
                        stop_event=stop_event,
                    ),
                    name=f"pipeline-{symbol}-{boundary.strftime('%H%M')}",
                )
                in_flight.add(task)
                task.add_done_callback(lambda t: _on_pipeline_done(t, in_flight))

            if stop_event.is_set():
                break

        except Exception as exc:
            logger.exception("scheduler iteration failed: %s — realigning", exc)
            await asyncio.sleep(5)

    # Graceful shutdown: cancel staggered pipelines instead of waiting minutes.
    if in_flight:
        logger.info(
            "scheduler stopping — cancelling %d in-flight pipeline(s)", len(in_flight),
        )
        for task in list(in_flight):
            task.cancel()
        await asyncio.gather(*in_flight, return_exceptions=True)
    logger.info("scheduler stopped")


def _on_pipeline_done(task: asyncio.Task[None], in_flight: set[asyncio.Task[None]]) -> None:
    """Reap a finished pipeline task; surface anything that escaped (it shouldn't)."""
    in_flight.discard(task)
    if task.cancelled():
        logger.warning("pipeline task %s was cancelled", task.get_name())
        return
    exc = task.exception()
    if exc is not None:
        # run_analysis_pipeline never raises by contract — this is a bug alarm.
        logger.critical(
            "pipeline task %s leaked an exception (contract violation): %s",
            task.get_name(), exc, exc_info=exc,
        )
