"""
Sentinel Trader — Analysis Pipeline.

One full audited cycle for one symbol:

    PipelineRun(running) → MarketSnapshot → FeaturePacket → [pre-gate]
        → AiDecision → RiskEngine.evaluate → [if approved] Broker
        → ExecutionAttempt / Order / Trade rows → PipelineRun(final)

Fail-safety contract:
- ``run_analysis_pipeline`` **never raises**. Every failure path is caught,
  logged as an ``Event`` row, and the pipeline run is finalized with
  outcome ``error`` or ``skipped``. Any ambiguity resolves to no action.
- The AI never talks to the broker: execution happens only from an
  approved ``EngineVerdict`` whose size/SL were recomputed by the engine.
- Every phase transition is persisted, so the audit chain explains every
  cycle — including the ones that did nothing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import aiosqlite

from sentinel.ai import AiClient
from sentinel.ai import AiDecision as AiDecisionContract
from sentinel.ai.reflection import run_veto_postmortem
from sentinel.config import Settings, get_settings
from sentinel.core.portfolio import load_portfolio_state
from sentinel.data.features import FeatureError, FeaturePacket, compute_features
from sentinel.data.market import MarketDataClient, MarketDataError, check_snapshot_sanity
from sentinel.data.news import NewsClient
from sentinel.exec.broker import Broker, OpenPositionRequest, OpenPositionResult
from sentinel.risk.engine import EngineVerdict, RiskEngine
from sentinel.risk.killswitch import KillSwitch
from sentinel.store import get_connection, repo
from sentinel.store.models import AiDecision as AiDecisionRow
from sentinel.store.models import (
    Event,
    ExecutionAttempt,
    FeatureSnapshot,
    Order,
    PipelineRun,
    Trade,
)

logger = logging.getLogger(__name__)

#: Async callback for operator notifications (Telegram). Must never raise.
AlertFn = Callable[[str], Awaitable[None]]

#: Serializes the risk→exec section across concurrent per-symbol pipelines.
#: Without it, two simultaneous approvals can both pass the max-positions /
#: drawdown gates before either position exists (read-then-act race).
#: Also taken by the position manager's orphan-adoption pass so it never
#: observes a broker fill whose trade row hasn't been persisted yet.
EXECUTION_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineContext:
    """Long-lived singletons shared by every pipeline run and loop.

    Built once at startup by ``sentinel.main`` and passed explicitly —
    no module-level mutable state.
    """

    market: MarketDataClient
    ai: AiClient
    risk: RiskEngine
    killswitch: KillSwitch
    broker: Broker | None = None          # None during Phase 1 (scanner only)
    news: NewsClient | None = None        # None = news layer disabled
    settings: Settings | None = None
    alert: AlertFn | None = None          # operator notifications (best-effort)

    @property
    def cfg(self) -> Settings:
        return self.settings or get_settings()

    async def notify(self, message: str) -> None:
        """Fire an operator alert; failures are logged, never propagated."""
        if self.alert is None:
            return
        try:
            await self.alert(message)
        except Exception as exc:  # noqa: BLE001 — alerts must never break the pipeline
            logger.error("alert delivery failed: %s", exc)


class _PipelineAbort(Exception):
    """Internal control flow: cleanly finalize the run with a given outcome."""

    def __init__(self, outcome: str, phase: str, reason: str) -> None:
        super().__init__(reason)
        self.outcome = outcome    # 'completed' | 'skipped' | 'error'
        self.phase = phase
        self.reason = reason


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_analysis_pipeline(ctx: PipelineContext, symbol: str, timeframe: str) -> None:
    """Run one full scan→decide→gate→execute cycle for one symbol.

    Never raises. All outcomes (completed / skipped / error) are persisted.
    """
    run = PipelineRun(
        symbol=symbol,
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        phase="scan",
        outcome="running",
    )
    logger.info("pipeline start: %s (%s candle close, run %s)", symbol, timeframe, run.id[:8])

    _crash_phase = "scan"
    try:
        async with get_connection() as db:
            await repo.insert_pipeline_run(db, run)
            try:
                await _execute_phases(ctx, db, run, symbol, timeframe)
                await repo.update_pipeline_run(db, run.id, "exec", "completed")
            except _PipelineAbort as abort:
                _crash_phase = abort.phase
                await repo.update_pipeline_run(db, run.id, abort.phase, abort.outcome)
                logger.info(
                    "pipeline %s for %s (%s): phase=%s — %s",
                    abort.outcome, symbol, run.id[:8], abort.phase, abort.reason,
                )
    except Exception as exc:
        logger.exception("pipeline run %s for %s crashed: %s", run.id[:8], symbol, exc)
        await _log_crash_best_effort(run.id, symbol, exc, _crash_phase)


async def _execute_phases(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    run: PipelineRun,
    symbol: str,
    timeframe: str,
) -> None:
    """The strict phase sequence. Raises ``_PipelineAbort`` to stop cleanly."""
    await _phase_pre_gate(ctx, db, run, symbol)

    features, recent_news = await _phase_scan(ctx, db, run, symbol, timeframe)

    decision, ai_row_id = await _phase_ai(ctx, db, run, symbol, features, recent_news)
    if decision.decision == "no_trade":
        raise _PipelineAbort("completed", "ai", "AI decided no_trade")

    # Portfolio gates and execution must be atomic across symbols: holding
    # the lock from state read to fill prevents over-opening past the caps.
    async with EXECUTION_LOCK:
        verdict, verdict_row_id = await _phase_risk(ctx, db, run, decision, features, ai_row_id)
        if not verdict.approved:
            if verdict.gates_failed:
                asyncio.create_task(
                    run_veto_postmortem(
                        ctx.ai,
                        symbol=symbol,
                        gate=verdict.gates_failed[0],
                        veto_reason=verdict.veto_reason or verdict.gates_failed[0],
                        decision=decision.decision,
                        confidence=decision.confidence,
                        features_json=features.to_json(),
                    )
                )
            raise _PipelineAbort("completed", "risk", f"vetoed: {verdict.veto_reason}")

        await _phase_exec(ctx, db, run, decision, verdict, verdict_row_id)


# ---------------------------------------------------------------------------
# Phase: pre-gate (cheap checks BEFORE any API spend)
# ---------------------------------------------------------------------------


async def _phase_pre_gate(
    ctx: PipelineContext, db: aiosqlite.Connection, run: PipelineRun, symbol: str,
) -> None:
    status = await ctx.killswitch.status(db)
    if status.halted:
        await _log_skip(db, run, symbol, f"halted: {status.reason}")
        raise _PipelineAbort("skipped", "pre_gate", f"halted: {status.reason}")

    if await repo.get_state(db, "paused") == "true":
        await _log_skip(db, run, symbol, "paused by admin")
        raise _PipelineAbort("skipped", "pre_gate", "paused by admin")

    budget = ctx.ai.get_budget_status()
    if budget["calls_remaining"] <= 0:
        await _log_skip(db, run, symbol, "AI daily budget exhausted")
        raise _PipelineAbort("skipped", "pre_gate", "AI daily budget exhausted")

    if budget.get("gemini_daily_exhausted"):
        await _log_skip(db, run, symbol, "Gemini daily quota exhausted (both keys)")
        raise _PipelineAbort("skipped", "pre_gate", "Gemini daily quota exhausted (both keys)")


# ---------------------------------------------------------------------------
# Phase: scan — market fetch + features
# ---------------------------------------------------------------------------


async def _phase_scan(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    run: PipelineRun,
    symbol: str,
    timeframe: str,
) -> tuple[FeaturePacket, list[str]]:
    """Fetch market data + news concurrently, compute features.

    Returns ``(features, recent_news)``. News is best-effort: an empty
    list on any failure — headlines enrich the AI call, never gate it.
    """
    await repo.update_pipeline_run(db, run.id, "scan", "running")

    news_task = (
        asyncio.create_task(ctx.news.get_headlines())
        if ctx.news is not None else None
    )

    try:
        snapshot = await ctx.market.fetch_market_snapshot(symbol)
    except MarketDataError as exc:
        if news_task is not None:
            news_task.cancel()
        await _log_error(db, run, symbol, "market_fetch", exc)
        raise _PipelineAbort("error", "scan", f"market fetch failed: {exc}") from exc

    recent_news: list[str] = []
    if news_task is not None:
        try:
            recent_news = await news_task   # NewsClient.get_headlines never raises
        except asyncio.CancelledError:
            pass

    sanity_failures = check_snapshot_sanity(snapshot, ctx.cfg)
    if sanity_failures:
        await _log_skip(db, run, symbol, f"data sanity: {'; '.join(sanity_failures)}")
        raise _PipelineAbort("skipped", "scan", f"data sanity failed: {sanity_failures}")

    try:
        features = compute_features(snapshot)
    except FeatureError as exc:
        await _log_skip(db, run, symbol, f"incomplete features: {exc}")
        raise _PipelineAbort("skipped", "scan", f"feature computation failed: {exc}") from exc

    await repo.insert_feature_snapshot(db, FeatureSnapshot(
        pipeline_run_id=run.id,
        symbol=symbol,
        features_json=features.to_json(),
    ))
    return features, recent_news


# ---------------------------------------------------------------------------
# Phase: ai — Claude decision
# ---------------------------------------------------------------------------


async def _phase_ai(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    run: PipelineRun,
    symbol: str,
    features: FeaturePacket,
    recent_news: list[str],
) -> tuple[AiDecisionContract, str]:
    """Request a decision and persist the full AI exchange. Returns (decision, row_id)."""
    await repo.update_pipeline_run(db, run.id, "ai", "running")

    open_trades = await repo.get_open_trades(db)
    active_positions = [
        {"symbol": t.symbol, "side": t.side, "entry_price": t.entry_price, "size": t.size}
        for t in open_trades
        if t.symbol == symbol
    ]
    recent = await repo.get_recent_decisions(db, symbol, limit=5)
    recent_decisions = [
        {"decision": d.decision, "confidence": d.confidence, "at": d.created_at}
        for d in recent
        if d.decision is not None
    ]

    lessons = await repo.get_recent_lessons(db, symbol, limit=3)
    past_lessons = [L.lesson_text for L in lessons]

    recent_vetoes = await repo.get_recent_vetoes(db, symbol, limit=5)

    decision, meta = await ctx.ai.request_decision(
        feature_dict=features.to_dict(),
        active_positions=active_positions,
        recent_decisions=recent_decisions,
        past_lessons=past_lessons,
        recent_news=recent_news,
        recent_vetoes=recent_vetoes or None,
    )

    raw_response = meta.get("raw_text") or (
        decision.model_dump_json() if decision is not None
        else json.dumps({"error": meta.get("error")})
    )
    ai_row = AiDecisionRow(
        pipeline_run_id=run.id,
        symbol=symbol,
        raw_response=raw_response,
        parsed_json=decision.model_dump_json() if decision is not None else None,
        decision=decision.decision if decision is not None else "error",
        confidence=decision.confidence if decision is not None else None,
        model_id=str(meta.get("model")),
        latency_ms=int(meta.get("latency_ms") or 0),
        input_tokens=int(meta.get("input_tokens") or 0),
        output_tokens=int(meta.get("output_tokens") or 0),
    )
    await repo.insert_ai_decision(db, ai_row)

    if decision is None:
        error = str(meta.get("error"))
        if error == "malformed_response":
            await ctx.killswitch.record_malformed_response(db, error)
        await _log_error(db, run, symbol, "ai_call", RuntimeError(error))
        raise _PipelineAbort("error", "ai", f"no usable AI decision: {error}")

    if decision.symbol != symbol:
        # Model answered for the wrong symbol — treat as malformed output.
        await ctx.killswitch.record_malformed_response(
            db, f"symbol mismatch: got {decision.symbol}, expected {symbol}",
        )
        await _log_error(db, run, symbol, "ai_call",
                         RuntimeError(f"AI symbol mismatch: {decision.symbol} != {symbol}"))
        raise _PipelineAbort("error", "ai", "AI answered for the wrong symbol")

    await ctx.killswitch.record_valid_response(db)
    return decision, ai_row.id


# ---------------------------------------------------------------------------
# Phase: risk — deterministic verdict
# ---------------------------------------------------------------------------


async def _phase_risk(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    run: PipelineRun,
    decision: AiDecisionContract,
    features: FeaturePacket,
    ai_row_id: str,
) -> tuple[EngineVerdict, str]:
    await repo.update_pipeline_run(db, run.id, "risk", "running")

    try:
        portfolio = await load_portfolio_state(db, ctx.broker)
        precision = ctx.market.get_precision_spec(decision.symbol)
        verdict = await ctx.risk.evaluate(db, decision, features, portfolio, precision)
    except Exception as exc:
        await _log_error(db, run, decision.symbol, "risk_engine", exc)
        raise _PipelineAbort("error", "risk", f"risk evaluation failed: {exc}") from exc

    verdict_row_id = await ctx.risk.persist_verdict(
        db, verdict,
        pipeline_run_id=run.id,
        ai_decision_id=ai_row_id,
        symbol=decision.symbol,
    )
    return verdict, verdict_row_id


# ---------------------------------------------------------------------------
# Phase: exec — broker attempt
# ---------------------------------------------------------------------------


async def _phase_exec(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    run: PipelineRun,
    decision: AiDecisionContract,
    verdict: EngineVerdict,
    verdict_row_id: str,
) -> None:
    await repo.update_pipeline_run(db, run.id, "exec", "running")
    if verdict.sizing is None or verdict.entry_price is None:
        raise _PipelineAbort("error", "exec", "approved verdict missing sizing (engine bug)")

    if ctx.broker is None:
        await repo.insert_event(db, Event(
            event_type="execution_skipped",
            severity="info",
            message=f"Approved {decision.decision} on {decision.symbol} not executed "
                    "(no broker configured — Phase 1 scanner mode)",
            context_json=json.dumps({"pipeline_run_id": run.id}),
        ))
        return

    # Idempotency guard (spec §12): never double-enter on the same symbol.
    open_trades = await repo.get_open_trades(db)
    if any(t.symbol == decision.symbol for t in open_trades):
        raise _PipelineAbort(
            "skipped", "exec", f"position already open on {decision.symbol} (idempotency guard)"
        )

    request = OpenPositionRequest(
        symbol=decision.symbol,
        side=decision.decision,  # 'long' | 'short' guaranteed by the engine
        entry_type=decision.entry.type,
        limit_price=verdict.entry_price if decision.entry.type == "limit" else None,
        contracts=verdict.sizing.contracts,
        leverage=verdict.sizing.leverage,
        stop_loss_price=Decimal(str(decision.stop_loss_price)),
        take_profit_prices=tuple(Decimal(str(tp)) for tp in decision.take_profit_prices),
        pipeline_run_id=run.id,
    )

    attempt = ExecutionAttempt(
        pipeline_run_id=run.id,
        risk_verdict_id=verdict_row_id,
        broker_type=ctx.broker.name,
        request_json=json.dumps(request.to_dict()),
        status="error",
    )

    try:
        result: OpenPositionResult = await asyncio.wait_for(
            ctx.broker.open_position(request),
            timeout=float(ctx.cfg.exec_timeout_sec),
        )
    except TimeoutError:
        attempt.status = "timeout"
        attempt.error_message = f"broker call exceeded {ctx.cfg.exec_timeout_sec}s"
        await repo.insert_execution_attempt(db, attempt)
        await ctx.killswitch.record_execution_error(db, "broker timeout")
        await ctx.notify(
            f"EXECUTION TIMEOUT — {decision.symbol}\n"
            f"Broker call exceeded {ctx.cfg.exec_timeout_sec}s. "
            f"Kill-switch error counter incremented."
        )
        raise _PipelineAbort("error", "exec", "broker call timed out") from None
    except Exception as exc:
        attempt.status = "error"
        attempt.error_message = f"{type(exc).__name__}: {exc}"
        await repo.insert_execution_attempt(db, attempt)
        await ctx.killswitch.record_execution_error(db, str(exc))
        await ctx.notify(
            f"EXECUTION ERROR — {decision.symbol}\n"
            f"{type(exc).__name__}: {exc}"
        )
        raise _PipelineAbort("error", "exec", f"broker call failed: {exc}") from exc

    attempt.response_json = json.dumps(result.to_dict())
    attempt.status = "success" if result.success else "error"
    attempt.error_message = result.error_message
    await repo.insert_execution_attempt(db, attempt)

    if not result.success:
        await ctx.killswitch.record_execution_error(db, result.error_message or "rejected")
        await ctx.notify(
            f"ORDER REJECTED — {decision.symbol}\n"
            f"Broker rejected: {result.error_message}"
        )
        raise _PipelineAbort("error", "exec", f"broker rejected: {result.error_message}")

    await ctx.killswitch.record_execution_success(db)
    await _persist_fills(db, run, decision, verdict, result)

    fill = result.fill_price or verdict.entry_price
    logger.info(
        "EXECUTED %s %s: contracts=%s entry=%s broker=%s",
        decision.symbol, decision.decision,
        verdict.sizing.contracts, fill, ctx.broker.name,
    )
    await ctx.notify(
        f"Trade opened: {decision.decision.upper()} {decision.symbol}\n"
        f"size: {verdict.sizing.contracts} @ {fill}\n"
        f"SL: {decision.stop_loss_price}  TP: {decision.take_profit_prices}\n"
        f"risk: {verdict.sizing.actual_risk_pct:.3f}%  lev: {verdict.sizing.leverage}x"
    )


async def _persist_fills(
    db: aiosqlite.Connection,
    run: PipelineRun,
    decision: AiDecisionContract,
    verdict: EngineVerdict,
    result: OpenPositionResult,
) -> None:
    """Write Order rows for every placed order plus the open Trade row."""
    if verdict.sizing is None or verdict.entry_price is None:
        raise _PipelineAbort("error", "persist", "approved verdict missing sizing (engine bug)")

    placed = [o for o in (result.entry_order, result.sl_order, *result.tp_orders) if o is not None]
    for broker_order in placed:
        await repo.insert_order(db, Order(
            id=broker_order.id,
            pipeline_run_id=run.id,
            symbol=broker_order.symbol,
            side=broker_order.side,
            order_type=broker_order.order_type,
            purpose=broker_order.purpose,
            price=float(broker_order.price) if broker_order.price is not None else None,
            size=float(broker_order.amount),
            status=broker_order.status,
            fill_price=float(broker_order.fill_price) if broker_order.fill_price else None,
            fill_time=broker_order.fill_time,
        ))

    entry_price = result.fill_price or verdict.entry_price
    await repo.insert_trade(db, Trade(
        pipeline_run_id=run.id,
        symbol=decision.symbol,
        side=decision.decision,
        entry_price=float(entry_price),
        size=float(verdict.sizing.contracts),
        leverage=verdict.sizing.leverage,
        status="open",
    ))


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


async def _log_skip(
    db: aiosqlite.Connection, run: PipelineRun, symbol: str, reason: str,
) -> None:
    logger.info("skip %s (%s): %s", symbol, run.id[:8], reason)
    await repo.insert_event(db, Event(
        event_type="pipeline_skip",
        severity="info",
        message=f"Pipeline skipped for {symbol}: {reason}",
        context_json=json.dumps({"pipeline_run_id": run.id}),
    ))


async def _log_error(
    db: aiosqlite.Connection, run: PipelineRun, symbol: str, where: str, exc: BaseException,
) -> None:
    logger.error("pipeline error in %s for %s (%s): %s", where, symbol, run.id[:8], exc)
    await repo.insert_event(db, Event(
        event_type="error",
        severity="error",
        message=f"Pipeline error in {where} for {symbol}: {exc}",
        context_json=json.dumps({
            "pipeline_run_id": run.id,
            "where": where,
            "exception_type": type(exc).__name__,
        }),
    ))


async def _log_crash_best_effort(
    run_id: str, symbol: str, exc: BaseException, phase: str = "unknown",
) -> None:
    """Last-resort error logging on a fresh connection (the original may be dead)."""
    try:
        async with get_connection() as db:
            await repo.insert_event(db, Event(
                event_type="error",
                severity="critical",
                message=f"Unhandled pipeline exception for {symbol}: {exc}",
                context_json=json.dumps({
                    "pipeline_run_id": run_id,
                    "exception_type": type(exc).__name__,
                }),
            ))
            await repo.update_pipeline_run(db, run_id, phase, "error")
    except Exception:
        logger.exception("failed to persist crash event for run %s", run_id[:8])
