"""
Sentinel Trader — Daily Performance Report.

Background task that fires at midnight UTC and sends a summary
of the day's trading activity to Telegram.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sentinel.core.pipeline import PipelineContext
from sentinel.store import get_connection, repo

logger = logging.getLogger(__name__)


async def run_daily_report(ctx: PipelineContext, stop_event: asyncio.Event) -> None:
    """Send a daily performance report at midnight UTC. Runs until stop_event."""
    logger.info("daily report task started")

    while not stop_event.is_set():
        try:
            # Calculate seconds until next midnight UTC
            now = datetime.now(UTC)
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
            wait_sec = (tomorrow - now).total_seconds()

            logger.debug("daily report sleeping %.0fs until %s", wait_sec, tomorrow.isoformat())

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_sec)
                break  # stop_event set
            except TimeoutError:
                pass  # midnight reached

            await _send_report(ctx)

        except Exception as exc:
            logger.exception("daily report iteration failed: %s", exc)
            await asyncio.sleep(60)

    logger.info("daily report task stopped")


async def _send_report(ctx: PipelineContext) -> None:
    """Build and send the daily report."""
    now = datetime.now(UTC)
    day_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    since_iso = day_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    report_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    async with get_connection() as db:
        trades = await repo.get_recent_trades(db, limit=500)
        equity_raw = await repo.get_state(db, "paper_equity")
        halt = await ctx.killswitch.status(db)

    # Filter to trades closed yesterday
    closed_today = [
        t for t in trades
        if t.status == "closed"
        and t.close_time is not None
        and t.close_time >= since_iso
    ]

    # Use net PnL (after fees/funding) when available; fall back to gross for
    # legacy rows written before cost accounting was added.
    def _net(t: object) -> float:
        np = getattr(t, "net_pnl", None)
        return np if np is not None else (getattr(t, "realized_pnl", 0.0) or 0.0)

    total_pnl = sum(_net(t) for t in closed_today)
    wins = sum(1 for t in closed_today if _net(t) > 0)
    losses = sum(1 for t in closed_today if _net(t) <= 0)
    total = len(closed_today)
    win_rate = (wins / total * 100) if total > 0 else 0.0

    best = max((_net(t) for t in closed_today), default=0.0)
    worst = min((_net(t) for t in closed_today), default=0.0)

    equity = float(equity_raw) if equity_raw else 10000.0
    budget = ctx.ai.get_budget_status()
    status_str = "🔴 HALTED" if halt.halted else "✅ Running"

    lines = [
        f"📊 DAILY REPORT — {report_date}",
        "",
        f"💰 Equity: ${equity:,.2f} USDT",
    ]

    if total > 0:
        lines.extend([
            f"📈 Trades: {total} ({wins}W / {losses}L — {win_rate:.1f}% win rate)",
            f"💵 PnL: {'+' if total_pnl >= 0 else ''}{total_pnl:.4f} USDT",
            f"🏆 Best: {'+' if best >= 0 else ''}{best:.4f} USDT",
            f"📉 Worst: {'+' if worst >= 0 else ''}{worst:.4f} USDT",
        ])
    else:
        lines.append("📈 Trades: 0 (no trades closed)")

    lines.extend([
        "",
        f"🧠 AI calls: {budget['calls_today']} | remaining: {budget['calls_remaining']}",
        f"⚡ Status: {status_str}",
    ])

    # Confidence calibration — the running edge tie-breaker (best-effort).
    try:
        from sentinel.analysis import format_daily_summary
        lines.append(await format_daily_summary())
    except Exception as exc:  # noqa: BLE001 — analysis must never break the report
        logger.warning("calibration summary failed: %s", exc)

    await ctx.notify("\n".join(lines))
    logger.info("daily report sent for %s", report_date)
