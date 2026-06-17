"""
Sentinel Trader — Telegram Admin Bot.

Full command & control interface for the operator:

Commands:
  /help          — list all available commands
  /status        — equity, 24h PnL, open positions, halt state
  /scan          — force an immediate pipeline run for all symbols
  /trades        — show last 10 trades with PnL
  /pnl           — daily + weekly + all-time PnL breakdown
  /ai            — AI budget status (calls, malformed, model)
  /positions     — detailed open positions with unrealized PnL
  /setconfidence — adjust confidence threshold live
  /lessons       — view the AI's internal memory bank (past lessons)
  /killswitch    — engage emergency halt
  /resume        — release kill switch

Security: every inbound message is filtered against the single whitelisted
``telegram_admin_chat_id``. Messages from anyone else are silently ignored.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.types import Message
from aiohttp.resolver import AsyncResolver

from sentinel.config import Settings, get_settings
from sentinel.net.dns import GOOGLE_DNS_SERVERS
from sentinel.risk.killswitch import KillSwitch
from sentinel.store import get_connection, repo

if TYPE_CHECKING:
    from sentinel.core.pipeline import PipelineContext

logger = logging.getLogger(__name__)

_TELEGRAM_MAX_LEN: Final[int] = 4096
_TRUNCATION_SUFFIX: Final[str] = "\n… (truncated)"
_TELEGRAM_BODY_MAX: Final[int] = _TELEGRAM_MAX_LEN - len(_TRUNCATION_SUFFIX)
_TELEGRAM_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{20,}$")
_PLACEHOLDER_TOKENS = frozenset({"", "dummy", "disabled", "none", "your_telegram_bot_token_here"})


def telegram_token_valid(token: str) -> bool:
    """True when ``token`` looks like a real Telegram Bot API token."""
    t = token.strip()
    if t.lower() in _PLACEHOLDER_TOKENS:
        return False
    return bool(_TELEGRAM_TOKEN_RE.match(t))


class _GoogleDnsAiohttpSession(AiohttpSession):
    """aiogram HTTP session that resolves via Google public DNS."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._connector_init = {
            **self._connector_init,
            "resolver": AsyncResolver(nameservers=list(GOOGLE_DNS_SERVERS)),
            "family": 0,
        }


class DisabledAdminBot:
    """No-op admin when Telegram is not configured."""

    def set_context(self, ctx: PipelineContext) -> None:
        pass

    def start(self) -> None:
        logger.info("Telegram admin disabled (set TELEGRAM_BOT_TOKEN to enable)")

    async def stop(self) -> None:
        pass

    async def send_alert(self, message: str) -> None:
        logger.debug("telegram disabled — alert skipped: %s", message[:200])


def create_admin_bot(
    settings: Settings | None = None,
    killswitch: KillSwitch | None = None,
) -> AdminBot | DisabledAdminBot:
    """Return a live admin bot or a no-op stub when the token is missing/invalid."""
    cfg = settings or get_settings()
    if telegram_token_valid(cfg.telegram_bot_token):
        return AdminBot(cfg, killswitch)
    return DisabledAdminBot()


class AdminBot:
    """aiogram polling bot bound to a single whitelisted admin chat."""

    def __init__(
        self,
        settings: Settings | None = None,
        killswitch: KillSwitch | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._killswitch: KillSwitch = killswitch or KillSwitch(self._settings)
        self._admin_chat_id: int = self._settings.telegram_admin_chat_id
        self._ctx: PipelineContext | None = None

        self._api_session = _GoogleDnsAiohttpSession()
        self._bot = Bot(token=self._settings.telegram_bot_token, session=self._api_session)
        self._dispatcher = Dispatcher()
        self._polling_task: asyncio.Task[None] | None = None
        self._scan_tasks: set[asyncio.Task[None]] = set()

        router = Router(name="admin")
        # Hard whitelist: anything not from the admin chat never reaches a handler.
        router.message.filter(F.chat.id == self._admin_chat_id)
        router.message.register(self._cmd_help, Command("help"))
        router.message.register(self._cmd_status, Command("status"))
        router.message.register(self._cmd_scan, Command("scan"))
        router.message.register(self._cmd_trades, Command("trades"))
        router.message.register(self._cmd_pnl, Command("pnl"))
        router.message.register(self._cmd_ai, Command("ai"))
        router.message.register(self._cmd_positions, Command("positions"))
        router.message.register(self._cmd_setconfidence, Command("setconfidence"))
        router.message.register(self._cmd_lessons, Command("lessons"))
        router.message.register(self._cmd_killswitch, Command("killswitch"))
        router.message.register(self._cmd_resume, Command("resume"))
        self._dispatcher.include_router(router)

    # ------------------------------------------------------------------
    # Context wiring
    # ------------------------------------------------------------------

    def set_context(self, ctx: PipelineContext) -> None:
        """Wire the bot to the full pipeline context (called after ctx is built)."""
        self._ctx = ctx

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start long-polling in a background task (non-blocking)."""
        if self._polling_task is not None:
            logger.warning("admin bot already started")
            return
        self._polling_task = asyncio.create_task(
            self._poll_with_clean_start(),
            name="admin-bot-polling",
        )
        logger.info("admin bot polling started (admin chat %d)", self._admin_chat_id)

    async def _poll_with_clean_start(self) -> None:
        """Clear any stale webhook/long-poll session, then start polling."""
        try:
            await self._bot.delete_webhook(drop_pending_updates=True)
            logger.info("webhook cleared before polling")
        except Exception as exc:
            logger.warning("delete_webhook failed (proceeding): %s", exc)
        await self._dispatcher.start_polling(self._bot, handle_signals=False)

    async def stop(self) -> None:
        """Stop polling and release the HTTP session. Never raises."""
        if self._scan_tasks:
            logger.info("cancelling %d manual /scan pipeline(s)", len(self._scan_tasks))
            for task in list(self._scan_tasks):
                task.cancel()
            await asyncio.gather(*self._scan_tasks, return_exceptions=True)
            self._scan_tasks.clear()
        try:
            if self._polling_task is not None:
                await self._dispatcher.stop_polling()
                self._polling_task.cancel()
                try:
                    await self._polling_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                self._polling_task = None
        except Exception as exc:  # noqa: BLE001 — shutdown must not propagate
            logger.warning("error stopping admin bot polling: %s", exc)
        finally:
            try:
                await self._bot.session.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("error closing bot session: %s", exc)
        logger.info("admin bot stopped")

    # ------------------------------------------------------------------
    # Outbound alerts
    # ------------------------------------------------------------------

    async def send_alert(self, message: str) -> None:
        """Send a notification to the admin chat. Best-effort, never raises."""
        try:
            if len(message) > _TELEGRAM_MAX_LEN:
                message = message[:_TELEGRAM_BODY_MAX] + _TRUNCATION_SUFFIX
            await self._bot.send_message(self._admin_chat_id, message)
        except Exception as exc:  # noqa: BLE001 — alerting must never break callers
            logger.error("failed to send Telegram alert: %s", exc)

    async def _fetch_equity(self) -> float | None:
        """Live broker equity first; fall back to persisted paper_equity."""
        if self._ctx is not None and self._ctx.broker is not None:
            try:
                return float(await self._ctx.broker.fetch_equity())
            except Exception as exc:  # noqa: BLE001
                logger.warning("broker.fetch_equity failed, falling back to DB: %s", exc)
        try:
            async with get_connection() as db:
                equity_raw = await repo.get_state(db, "paper_equity")
            if equity_raw is not None:
                return float(equity_raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("paper_equity DB lookup failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Command handlers (admin-only by router filter)
    # ------------------------------------------------------------------

    async def _cmd_help(self, message: Message) -> None:
        """/help — list all available commands."""
        await message.answer(
            "🤖 Sentinel Trader — Commands\n"
            "\n"
            "/status — equity, PnL, positions, halt state\n"
            "/scan — force immediate scan (all symbols)\n"
            "/trades — last 10 trades with PnL\n"
            "/pnl — daily / weekly / all-time PnL\n"
            "/ai — AI budget & model status\n"
            "/positions — open positions detail\n"
            "/setconfidence <0.0-1.0> — adjust min confidence\n"
            "/lessons <symbol> — view AI memory for a coin\n"
            "/killswitch — emergency halt all trading\n"
            "/resume — release kill switch\n"
            "/help — this message"
        )

    async def _cmd_status(self, message: Message) -> None:
        """/status — equity, 24h realized PnL, open positions, halt state."""
        try:
            async with get_connection() as db:
                halt = await self._killswitch.status(db)
                open_trades = await repo.get_open_trades(db)
                recent_trades = await repo.get_recent_trades(db, limit=200)

            equity = await self._fetch_equity()

            cutoff = (datetime.now(UTC) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
            pnl_24h = sum(
                trade.realized_pnl or 0.0
                for trade in recent_trades
                if trade.status == "closed"
                and trade.close_time is not None
                and trade.close_time >= cutoff
            )
            closed_24h = sum(
                1 for trade in recent_trades
                if trade.status == "closed"
                and trade.close_time is not None
                and trade.close_time >= cutoff
            )

            lines = ["📊 Sentinel Trader — Status", ""]
            if halt.halted:
                lines.append(f"🔴 State: HALTED — {halt.reason}")
            else:
                lines.append("🟢 State: Running")
            if equity is not None:
                lines.append(f"💰 Equity: ${equity:,.2f} USDT")
            lines.append(f"📈 24h PnL: {pnl_24h:+.4f} USDT ({closed_24h} closed)")
            lines.append(f"⚙️ Confidence: {self._settings.confidence_threshold}")
            lines.append("")

            if open_trades:
                lines.append(f"📌 Open positions ({len(open_trades)}):")
                for trade in open_trades:
                    lines.append(
                        f"  {trade.symbol} {trade.side} size={trade.size} "
                        f"entry=${trade.entry_price:,.2f} lev={trade.leverage}x"
                    )
            else:
                lines.append("📌 Open positions: none")

            await message.answer("\n".join(lines)[:_TELEGRAM_MAX_LEN])
        except Exception as exc:  # noqa: BLE001
            logger.exception("/status failed: %s", exc)
            await message.answer(f"❌ status failed: {exc}")

    async def _cmd_scan(self, message: Message) -> None:
        """/scan — force an immediate pipeline run for all symbols."""
        if self._ctx is None:
            await message.answer("❌ Bot context not initialized yet.")
            return

        try:
            from sentinel.core.scheduler import run_staggered_pipeline, timeframe_label

            symbols = self._settings.scan_symbols
            timeframe = timeframe_label(self._settings.scan_interval_minutes)
            stagger = self._settings.scan_stagger_sec
            est_sec = int((len(symbols) - 1) * stagger)

            await message.answer(
                f"🔄 Scanning {len(symbols)} symbols (staggered {stagger:.0f}s, ~{est_sec}s total)"
            )

            for i, symbol in enumerate(symbols):
                task = asyncio.create_task(
                    run_staggered_pipeline(
                        self._ctx, symbol, timeframe, delay=i * stagger,
                    ),
                    name=f"manual-scan-{symbol}",
                )
                self._scan_tasks.add(task)
                task.add_done_callback(self._scan_tasks.discard)

            logger.info("manual scan triggered via /scan for %s", symbols)
        except Exception as exc:  # noqa: BLE001
            logger.exception("/scan failed: %s", exc)
            await message.answer(f"❌ scan failed: {exc}")

    async def _cmd_trades(self, message: Message) -> None:
        """/trades — show last 10 trades with PnL."""
        try:
            async with get_connection() as db:
                trades = await repo.get_recent_trades(db, limit=10)

            if not trades:
                await message.answer("📋 No trades yet.")
                return

            lines = ["📋 Recent Trades (last 10)", ""]
            for t in trades:
                pnl_str = f"{t.realized_pnl:+.4f}" if t.realized_pnl is not None else "open"
                emoji = "🟢" if (t.realized_pnl or 0) > 0 else "🔴" if t.realized_pnl is not None else "⏳"
                reason = f" [{t.close_reason}]" if t.close_reason else ""
                lines.append(
                    f"{emoji} {t.symbol} {t.side} "
                    f"${t.entry_price:,.2f} → "
                    f"{'$' + f'{t.exit_price:,.2f}' if t.exit_price else '...'} "
                    f"PnL: {pnl_str}{reason}"
                )

            await message.answer("\n".join(lines)[:_TELEGRAM_MAX_LEN])
        except Exception as exc:  # noqa: BLE001
            logger.exception("/trades failed: %s", exc)
            await message.answer(f"❌ trades failed: {exc}")

    async def _cmd_pnl(self, message: Message) -> None:
        """/pnl — daily + weekly + all-time PnL breakdown."""
        try:
            now = datetime.now(UTC)
            day_cutoff = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
            week_cutoff = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

            async with get_connection() as db:
                day_stats = await repo.get_closed_pnl_stats(db, day_cutoff)
                week_stats = await repo.get_closed_pnl_stats(db, week_cutoff)
                all_stats = await repo.get_closed_pnl_stats(db)

            equity = await self._fetch_equity()
            if equity is None:
                equity = 10000.0

            def _fmt(stats: dict[str, float | int]) -> str:
                w, l = int(stats["wins"]), int(stats["losses"])
                total = w + l
                wr = f"{w / total * 100:.0f}%" if total > 0 else "N/A"
                return (
                    f"{float(stats['total_pnl']):+.4f} USDT | "
                    f"{total} trades ({w}W/{l}L) | WR: {wr}"
                )

            lines = [
                "💹 PnL Breakdown", "",
                f"💰 Equity: ${equity:,.2f} USDT", "",
                f"📅 24h:     {_fmt(day_stats)}",
                f"📆 7d:      {_fmt(week_stats)}",
                f"📊 All-time: {_fmt(all_stats)}",
            ]

            await message.answer("\n".join(lines)[:_TELEGRAM_MAX_LEN])
        except Exception as exc:  # noqa: BLE001
            logger.exception("/pnl failed: %s", exc)
            await message.answer(f"❌ pnl failed: {exc}")

    async def _cmd_ai(self, message: Message) -> None:
        """/ai — AI budget status."""
        if self._ctx is None:
            await message.answer("❌ Bot context not initialized yet.")
            return

        try:
            budget = self._ctx.ai.get_budget_status()
            lines = [
                "🧠 AI Status", "",
                f"Provider: {self._settings.ai_provider}",
                f"Model: {self._settings.ai_model}",
                f"Calls today: {budget['calls_today']}",
                f"Budget remaining: {budget['calls_remaining']}",
                f"Max daily calls: {self._settings.max_ai_calls_per_day}",
                f"Consecutive malformed: {budget['consecutive_malformed']}",
                f"Budget date: {budget['budget_date']}",
            ]
            if self._settings.ai_provider == "dual":
                lines.append(f"NVIDIA fallbacks today: {budget.get('nvidia_fallbacks_today', 0)}")
            if self._settings.ai_provider == "gemini_dual":
                lines.append(
                    f"Gemini key 1 calls: {budget.get('gemini_key1_calls_today', 0)}"
                    f"{f' / {self._settings.gemini_key1_max_daily_calls}' if self._settings.gemini_key1_max_daily_calls else ''}"
                )
                lines.append(
                    f"Gemini key 2 calls: {budget.get('gemini_key2_calls_today', 0)}"
                    f"{f' / {self._settings.gemini_key2_max_daily_calls}' if self._settings.gemini_key2_max_daily_calls else ''}"
                )
            await message.answer("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            logger.exception("/ai failed: %s", exc)
            await message.answer(f"❌ ai failed: {exc}")

    async def _cmd_positions(self, message: Message) -> None:
        """/positions — detailed open positions with unrealized PnL."""
        if self._ctx is None or self._ctx.broker is None:
            await message.answer("❌ Broker not available.")
            return

        try:
            broker_positions = await self._ctx.broker.fetch_positions()
            async with get_connection() as db:
                open_trades = await repo.get_open_trades(db)

            if not broker_positions and not open_trades:
                await message.answer("📭 No open positions.")
                return

            lines = ["📌 Open Positions", ""]

            for pos in broker_positions:
                pnl_str = f"{float(pos.unrealized_pnl):+.4f}" if pos.unrealized_pnl else "N/A"
                pnl_emoji = "🟢" if (pos.unrealized_pnl or 0) > 0 else "🔴"
                lines.append(
                    f"{pnl_emoji} {pos.symbol} {pos.side}\n"
                    f"  Entry: ${float(pos.entry_price):,.2f}\n"
                    f"  Size: {float(pos.contracts)} contracts\n"
                    f"  Leverage: {pos.leverage}x\n"
                    f"  Unrealized PnL: {pnl_str} USDT"
                )
                lines.append("")

            # Show local trades without broker positions (edge case)
            broker_symbols = {p.symbol for p in broker_positions}
            orphan_trades = [t for t in open_trades if t.symbol not in broker_symbols]
            if orphan_trades:
                lines.append("⚠️ Local-only trades (no broker position):")
                for t in orphan_trades:
                    lines.append(f"  {t.symbol} {t.side} entry=${t.entry_price:,.2f}")

            await message.answer("\n".join(lines)[:_TELEGRAM_MAX_LEN])
        except Exception as exc:  # noqa: BLE001
            logger.exception("/positions failed: %s", exc)
            await message.answer(f"❌ positions failed: {exc}")

    async def _cmd_setconfidence(self, message: Message) -> None:
        """/setconfidence <value> — adjust confidence threshold live."""
        try:
            parts = (message.text or "").split()
            if len(parts) < 2:
                await message.answer(
                    f"Current confidence threshold: {self._settings.confidence_threshold}\n"
                    "Usage: /setconfidence 0.5"
                )
                return

            new_val = float(parts[1])
            if not 0.0 <= new_val <= 1.0:
                await message.answer("❌ Value must be between 0.0 and 1.0")
                return

            old_val = self._settings.confidence_threshold
            self._settings.confidence_threshold = new_val
            await message.answer(
                f"✅ Confidence threshold updated: {old_val} → {new_val}\n"
                "(Takes effect on next scan cycle)"
            )
            logger.info("confidence threshold changed via Telegram: %s -> %s", old_val, new_val)
        except ValueError:
            await message.answer("❌ Invalid number. Usage: /setconfidence 0.5")
        except Exception as exc:  # noqa: BLE001
            logger.exception("/setconfidence failed: %s", exc)
            await message.answer(f"❌ setconfidence failed: {exc}")

    async def _cmd_lessons(self, message: Message) -> None:
        """/lessons <symbol> — view the AI's internal memory bank."""
        try:
            parts = (message.text or "").split()
            if len(parts) < 2:
                await message.answer("Usage: /lessons BTC/USDT")
                return

            symbol = parts[1].upper()
            if not symbol.endswith("/USDT"):
                symbol += "/USDT"

            async with get_connection() as db:
                lessons = await repo.get_recent_lessons(db, symbol, limit=5)

            if not lessons:
                await message.answer(f"📭 No lessons in memory for {symbol} yet.")
                return

            lines = [f"🧠 AI Memory Bank — {symbol}", ""]
            for i, L in enumerate(lessons, 1):
                pnl_str = f"{L.pnl_usdt:+.4f}"
                emoji = "🟢" if L.pnl_usdt > 0 else "🔴"
                date_str = L.created_at[:10]
                lines.append(f"{i}. [{date_str} | {emoji} {pnl_str}]")
                lines.append(f"   {L.lesson_text}")
                lines.append("")

            await message.answer("\n".join(lines)[:_TELEGRAM_MAX_LEN])
        except Exception as exc:  # noqa: BLE001
            logger.exception("/lessons failed: %s", exc)
            await message.answer(f"❌ lessons failed: {exc}")

    async def _cmd_killswitch(self, message: Message) -> None:
        """/killswitch — immediately halt all trading (persisted)."""
        try:
            async with get_connection() as db:
                if await self._killswitch.is_halted(db):
                    await message.answer("Already halted.")
                    return
                await self._killswitch.engage(
                    db, "manual /killswitch from admin",
                    context={"source": "telegram"},
                )
            await message.answer(
                "🛑 KILL SWITCH ENGAGED.\n"
                "All trading halted (flag persisted — survives restarts).\n"
                "Use /resume to release."
            )
            logger.critical("kill switch engaged via Telegram /killswitch")
        except Exception as exc:  # noqa: BLE001
            logger.exception("/killswitch failed: %s", exc)
            await message.answer(f"❌ killswitch FAILED: {exc} — check the host immediately.")

    async def _cmd_resume(self, message: Message) -> None:
        """/resume — release the kill switch and reset failure counters."""
        try:
            async with get_connection() as db:
                if not await self._killswitch.is_halted(db):
                    await message.answer("Not halted — nothing to resume.")
                    return
                await self._killswitch.release(db, released_by="telegram admin")
            await message.answer("✅ Resumed. Trading re-enabled, failure counters reset.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("/resume failed: %s", exc)
            await message.answer(f"❌ resume failed: {exc}")
