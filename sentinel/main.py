"""
Sentinel Trader — Supervisor (entry point).

Startup sequence (spec §11):
    1. Configure structured logging (console + rotating JSONL).
    2. Initialize SQLite (WAL, schema).
    3. Load CCXT markets with retry.
    4. Build the ``PipelineContext`` (market, AI, risk, kill switch, broker).
    5. Start the aiogram admin bot (background polling).
    6. Run the scan scheduler and the position-manager fast loop concurrently.
    7. On SIGINT/SIGTERM: signal both loops, await in-flight work, close every
       session (broker, exchange, bot), exit 0.

If the persisted kill switch is engaged at startup, the process still runs
(admin bot + position management) but the pipeline pre-gate skips all
trading until ``/resume`` — matching the spec's halted-mode behaviour.

Run with: ``python -m sentinel.main``
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
from datetime import UTC, datetime

from sentinel.admin import create_admin_bot
from sentinel.ai import AiClient
from sentinel.config import Settings, get_settings
from sentinel.net.dns import install_google_dns_resolver
from sentinel.core import (
    manage_open_positions,
    run_daily_report,
    run_scan_scheduler,
    run_watchdog,
)
from sentinel.core.pipeline import PipelineContext
from sentinel.data.market import MarketDataClient
from sentinel.data.news import NewsClient
from sentinel.exec.broker import Broker
from sentinel.exec.paper import PaperBroker
from sentinel.risk.engine import RiskEngine
from sentinel.risk.killswitch import KillSwitch
from sentinel.store import init_db

logger = logging.getLogger("sentinel.main")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class _JsonLineFormatter(logging.Formatter):
    """One JSON object per line — machine-parseable, journald-friendly."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _setup_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s", "%H:%M:%S",
    ))
    root.addHandler(console)

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    jsonl = logging.handlers.TimedRotatingFileHandler(
        settings.log_dir / "sentinel.jsonl",
        when="midnight", utc=True, backupCount=14, encoding="utf-8",
    )
    jsonl.setFormatter(_JsonLineFormatter())
    root.addHandler(jsonl)

    # Third-party noise control.
    for noisy in ("aiogram", "ccxt", "aiosqlite", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Signal handling (POSIX + Windows)
# ---------------------------------------------------------------------------


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """Request graceful shutdown on SIGINT/SIGTERM, on any platform."""
    loop = asyncio.get_running_loop()
    force_exit = False

    def _request_stop(signame: str) -> None:
        nonlocal force_exit
        if force_exit:
            logger.critical("received %s again — forcing exit", signame)
            os._exit(130)
        force_exit = True
        logger.warning(
            "received %s — shutting down (press Ctrl+C again to force quit)", signame,
        )
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop, sig.name)
        except NotImplementedError:
            # Windows ProactorEventLoop: fall back to classic handlers that
            # hop back onto the loop thread-safely.
            signal.signal(
                sig,
                lambda s, _f: loop.call_soon_threadsafe(
                    _request_stop, signal.Signals(s).name
                ),
            )


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


async def main() -> int:
    settings = get_settings()
    install_google_dns_resolver()
    _setup_logging(settings)
    logger.info("sentinel-trader starting (symbols=%s, interval=%dmin)",
                settings.scan_symbols, settings.scan_interval_minutes)

    # 1. Database -----------------------------------------------------------
    await init_db()

    # 2. Market data (markets loaded with retry) -----------------------------
    market = MarketDataClient(settings)
    await market.connect()

    import argparse
    parser = argparse.ArgumentParser(description="Sentinel Trader")
    parser.add_argument("--live", action="store_true", help="Run with live MEXC execution broker")
    args = parser.parse_args()

    # 3. Broker + kill switch (emergency callback flattens the book) ---------
    if args.live:
        from sentinel.exec.mexc import MexcBroker
        broker: Broker = MexcBroker(settings)
    else:
        paper = PaperBroker(market, settings)
        # Rehydrate open positions so a restart doesn't force-close them as
        # 'manual' (markets are already loaded above).
        await paper.restore()
        broker = paper

    async def _emergency_flatten(reason: str) -> None:
        """Kill-switch callback: cancel everything, close everything."""
        logger.critical("emergency flatten: %s", reason)
        for position in await broker.fetch_positions():
            await broker.cancel_all_orders(position.symbol)
            await broker.close_position(position.symbol, reason=f"kill_switch: {reason}")

    killswitch = KillSwitch(settings, emergency_callback=_emergency_flatten)
    risk = RiskEngine(settings, killswitch)
    ai = AiClient(settings)
    news = NewsClient(settings)

    # 4. Admin bot ------------------------------------------------------------
    admin = create_admin_bot(settings, killswitch)

    # 5. Shared context --------------------------------------------------------
    ctx = PipelineContext(
        market=market,
        ai=ai,
        risk=risk,
        killswitch=killswitch,
        broker=broker,
        news=news,
        settings=settings,
        alert=admin.send_alert,
    )

    admin.set_context(ctx)

    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    admin.start()
    await admin.send_alert(
        f"sentinel-trader started\n"
        f"symbols: {', '.join(settings.scan_symbols)}\n"
        f"interval: {settings.scan_interval_minutes}min · broker: {broker.name}"
    )

    exit_code = 0
    try:
        # 6. Both loops run until stop_event is set; each is internally
        # fail-safe and only returns on shutdown.
        await asyncio.gather(
            run_scan_scheduler(ctx, stop_event),
            manage_open_positions(ctx, stop_event),
            run_daily_report(ctx, stop_event),
            run_watchdog(ctx, stop_event),
        )
    except Exception as exc:  # noqa: BLE001 — supervisor-level catastrophe
        logger.critical("supervisor crashed: %s", exc, exc_info=True)
        await admin.send_alert(f"CRITICAL: supervisor crashed: {exc}")
        exit_code = 1
    finally:
        # 7. Orderly teardown — every step independently fenced.
        logger.info("shutting down…")
        stop_event.set()
        await admin.send_alert("sentinel-trader shutting down")
        await admin.stop()
        await ai.close()
        await market.close()
        logger.info("shutdown complete")

    return exit_code


def run() -> None:
    """Console entry point (``python -m sentinel.main``)."""
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        # Second Ctrl+C during teardown — exit quietly.
        raise SystemExit(130) from None


if __name__ == "__main__":
    run()
