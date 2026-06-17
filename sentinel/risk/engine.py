"""
Sentinel Trader — Deterministic Risk Engine.

The single authority between an AI proposal and the broker. The model
**proposes**; this engine **disposes**. Every gate from spec §8 is checked
here, in a fixed order, and the first failure vetoes the trade. The engine
recomputes size, leverage, and risk itself — any size-related notion in the
AI output is ignored by construction (the contract has no such fields).

All price/equity arithmetic uses ``Decimal``. AI-supplied floats cross the
boundary exactly once, through ``to_decimal`` (str-mediated conversion).

Outputs are persisted to the ``risk_verdicts`` table, and every veto is
additionally logged to ``events`` — the audit chain must explain every
trade that *didn't* happen as well as every one that did.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC
from decimal import Decimal
from typing import Final

import aiosqlite

from sentinel.ai.contract import AiDecision
from sentinel.config import Settings, get_settings
from sentinel.data.features import FeaturePacket
from sentinel.data.market import PrecisionSpec
from sentinel.risk.killswitch import KillSwitch
from sentinel.risk.sizing import SizingError, SizingResult, compute_position_size, to_decimal
from sentinel.store.models import Event, RiskVerdict
from sentinel.store.repo import insert_event, insert_risk_verdict

logger = logging.getLogger(__name__)

_HUNDRED: Final[Decimal] = Decimal("100")



# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpenPosition:
    """Minimal view of an open position, as reconciled from the exchange."""

    symbol: str
    side: str            # 'long' | 'short'
    contracts: Decimal
    entry_price: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Snapshot of account state at evaluation time.

    Equity figures must come from the broker/reconciler (exchange is the
    source of truth), never from local bookkeeping alone.
    """

    equity_usdt: Decimal
    daily_start_equity: Decimal
    weekly_start_equity: Decimal
    open_positions: tuple[OpenPosition, ...] = field(default=())


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineVerdict:
    """Result of a full gate evaluation. Approval carries the computed size."""

    approved: bool
    veto_reason: str | None
    gates_passed: tuple[str, ...]
    gates_failed: tuple[str, ...]
    entry_price: Decimal | None          # resolved reference entry
    sizing: SizingResult | None          # present only on approval
    warnings: tuple[str, ...] = field(default=())

    @property
    def verdict(self) -> str:
        return "approve" if self.approved else "veto"


class _GateFailure(Exception):
    """Internal control flow: raised by a gate to veto with a reason."""

    def __init__(self, gate: str, reason: str) -> None:
        super().__init__(reason)
        self.gate = gate
        self.reason = reason


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RiskEngine:
    """All deterministic gates in one place (spec §8).

    Gate order (first failure vetoes):
        1.  kill_switch          — persisted halt flag
        2.  symbol_allowlist     — BTC/ETH/SOL only
        3.  decision_actionable  — long/short only (no_trade never reaches here)
        4.  confidence           — below threshold → veto
        4a. timeframe_alignment  — conflicting timeframes require confidence ≥ 0.75
        4b. rsi_extreme          — 1h RSI > 70 blocks longs; < 30 blocks shorts
        5.  max_positions        — concurrent position cap
        6.  one_per_symbol       — no doubling up
        7.  entry_validation     — limit price within band of market
        8.  sl_side              — SL strictly on the loss side
        9.  sl_distance          — within [min_sl_pct, max_sl_pct] of entry
        10. sl_atr_sanity        — SL distance ≥ 0.5× ATR(14) (flash-wick guard)
        11. tp_side              — every TP strictly on the profit side
        12. min_rr               — TP1/SL distance ratio ≥ threshold
        13. daily_drawdown       — breach also engages the kill switch (24h)
        14. weekly_drawdown      — breach also engages the kill switch (manual)
        15. post_loss_cooldown   — wait period after a losing close
        16. consecutive_losses   — cap before manual /resume
        17. sizing               — Decimal-exact size; min/max notional, leverage
    """

    def __init__(
        self,
        settings: Settings | None = None,
        killswitch: KillSwitch | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._killswitch: KillSwitch = killswitch or KillSwitch(self._settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        db: aiosqlite.Connection,
        decision: AiDecision,
        features: FeaturePacket,
        portfolio: PortfolioState,
        precision: PrecisionSpec,
    ) -> EngineVerdict:
        """Run every gate against an AI decision. Never raises on a veto.

        Raises only on structural bugs (e.g. ``SizingError`` from impossible
        inputs), which the pipeline must treat as an error event — fail safe.
        """
        passed: list[str] = []
        warnings: list[str] = []

        try:
            entry_price = await self._run_gates(
                db, decision, features, portfolio, passed, warnings,
            )
            sizing = self._gate_sizing(decision, entry_price, portfolio, precision, passed)
        except _GateFailure as failure:
            logger.warning(
                "VETO %s %s: gate=%s reason=%s",
                decision.symbol, decision.decision, failure.gate, failure.reason,
            )
            return EngineVerdict(
                approved=False,
                veto_reason=f"{failure.gate}: {failure.reason}",
                gates_passed=tuple(passed),
                gates_failed=(failure.gate,),
                entry_price=None,
                sizing=None,
                warnings=tuple(warnings),
            )

        logger.info(
            "APPROVE %s %s: entry=%s sl=%s contracts=%s risk=%.4f%% lev=%dx",
            decision.symbol, decision.decision, entry_price,
            decision.stop_loss_price, sizing.contracts,
            sizing.actual_risk_pct, sizing.leverage,
        )
        return EngineVerdict(
            approved=True,
            veto_reason=None,
            gates_passed=tuple(passed),
            gates_failed=(),
            entry_price=entry_price,
            sizing=sizing,
            warnings=tuple(warnings),
        )

    async def persist_verdict(
        self,
        db: aiosqlite.Connection,
        verdict: EngineVerdict,
        *,
        pipeline_run_id: str,
        ai_decision_id: str,
        symbol: str,
    ) -> str:
        """Write the verdict to ``risk_verdicts`` (and ``events`` on veto)."""
        sizing = verdict.sizing
        record = RiskVerdict(
            pipeline_run_id=pipeline_run_id,
            ai_decision_id=ai_decision_id,
            verdict=verdict.verdict,
            veto_reason=verdict.veto_reason,
            computed_size=float(sizing.contracts) if sizing else None,
            computed_leverage=sizing.leverage if sizing else None,
            risk_pct=float(sizing.actual_risk_pct) if sizing else None,
            gates_passed=json.dumps(list(verdict.gates_passed)),
            gates_failed=json.dumps(list(verdict.gates_failed)),
        )
        verdict_id = await insert_risk_verdict(db, record)

        if not verdict.approved:
            await insert_event(db, Event(
                event_type="risk_veto",
                severity="warning",
                message=f"Risk engine vetoed {symbol}: {verdict.veto_reason}",
                context_json=json.dumps({
                    "pipeline_run_id": pipeline_run_id,
                    "ai_decision_id": ai_decision_id,
                    "gates_passed": list(verdict.gates_passed),
                    "gates_failed": list(verdict.gates_failed),
                    "warnings": list(verdict.warnings),
                }),
            ))
        return verdict_id

    # ------------------------------------------------------------------
    # Gates 1–16 (everything except sizing)
    # ------------------------------------------------------------------

    async def _run_gates(
        self,
        db: aiosqlite.Connection,
        decision: AiDecision,
        features: FeaturePacket,
        portfolio: PortfolioState,
        passed: list[str],
        warnings: list[str],
    ) -> Decimal:
        """Run gates 1–16. Returns the resolved entry price on success."""
        cfg = self._settings

        # 1. Kill switch ------------------------------------------------
        status = await self._killswitch.status(db)
        if status.halted:
            raise _GateFailure("kill_switch", f"trading halted: {status.reason}")
        passed.append("kill_switch")

        # 2. Symbol allowlist -------------------------------------------
        if decision.symbol not in cfg.scan_symbols:
            raise _GateFailure("symbol_allowlist", f"symbol {decision.symbol!r} not allowed")
        if decision.symbol != features.symbol:
            raise _GateFailure(
                "symbol_allowlist",
                f"decision symbol {decision.symbol} != feature symbol {features.symbol}",
            )
        passed.append("symbol_allowlist")

        # 3. Actionable decision ----------------------------------------
        if decision.decision not in ("long", "short"):
            raise _GateFailure("decision_actionable", f"decision {decision.decision!r} is not actionable")
        passed.append("decision_actionable")
        is_long = decision.decision == "long"

        # 4. Confidence ---------------------------------------------------
        if decision.confidence < cfg.confidence_threshold:
            raise _GateFailure(
                "confidence",
                f"confidence {decision.confidence:.2f} < threshold {cfg.confidence_threshold:.2f}",
            )
        passed.append("confidence")

        # 4a. Timeframe alignment — conflicting timeframes require higher conviction
        if decision.timeframe_alignment == "conflicting":
            required = 0.75
            if decision.confidence < required:
                raise _GateFailure(
                    "timeframe_conflict",
                    f"conflicting timeframes require confidence ≥ {required:.2f}, "
                    f"got {decision.confidence:.2f}",
                )
            warnings.append("timeframe_alignment=conflicting — elevated confidence threshold applied")
        passed.append("timeframe_alignment")

        # 4b. RSI extreme — don't chase overbought longs or oversold shorts -------
        tf_1h = getattr(features, "timeframes", {}).get("1h")
        if tf_1h is not None:
            rsi_1h = getattr(tf_1h, "rsi_14", None)
            if rsi_1h is not None:
                if is_long and rsi_1h > cfg.rsi_overbought_threshold:
                    raise _GateFailure(
                        "rsi_extreme",
                        f"long blocked: 1h RSI {rsi_1h:.1f} > overbought {cfg.rsi_overbought_threshold}",
                    )
                if not is_long and rsi_1h < cfg.rsi_oversold_threshold:
                    raise _GateFailure(
                        "rsi_extreme",
                        f"short blocked: 1h RSI {rsi_1h:.1f} < oversold {cfg.rsi_oversold_threshold}",
                    )
        passed.append("rsi_extreme")

        # 5. Max concurrent positions ------------------------------------
        if len(portfolio.open_positions) >= cfg.max_concurrent_positions:
            raise _GateFailure(
                "max_positions",
                f"{len(portfolio.open_positions)} open >= cap {cfg.max_concurrent_positions}",
            )
        passed.append("max_positions")

        # 6. One position per symbol ---------------------------------------
        for pos in portfolio.open_positions:
            if pos.symbol == decision.symbol:
                raise _GateFailure(
                    "one_per_symbol",
                    f"position already open on {decision.symbol} ({pos.side} {pos.contracts})",
                )
        passed.append("one_per_symbol")

        # 7. Entry validation + entry price resolution ----------------------
        current_price = to_decimal(features.current_price, "current_price")
        if decision.entry.type == "limit":
            limit_price = to_decimal(decision.entry.limit_price, "limit_price")
            if limit_price <= 0:
                raise _GateFailure("entry_validation", f"non-positive limit price {limit_price}")
            deviation_pct = abs(limit_price - current_price) / current_price * _HUNDRED
            max_dev = Decimal(str(cfg.limit_entry_max_deviation_pct))
            if deviation_pct > max_dev:
                raise _GateFailure(
                    "entry_validation",
                    f"limit {limit_price} deviates {deviation_pct:.3f}% from market "
                    f"{current_price} (max {max_dev}%)",
                )
            entry_price = limit_price
        else:
            entry_price = current_price
        passed.append("entry_validation")

        # 8. SL side -----------------------------------------------------------
        sl = to_decimal(decision.stop_loss_price, "stop_loss_price")
        if sl <= 0:
            raise _GateFailure("sl_side", f"stop-loss must be positive, got {sl}")
        if is_long and sl >= entry_price:
            raise _GateFailure("sl_side", f"long SL {sl} must be below entry {entry_price}")
        if not is_long and sl <= entry_price:
            raise _GateFailure("sl_side", f"short SL {sl} must be above entry {entry_price}")
        passed.append("sl_side")

        # 9. SL distance range --------------------------------------------------
        sl_distance = abs(entry_price - sl)
        sl_distance_pct = sl_distance / entry_price * _HUNDRED
        min_sl, max_sl = Decimal(str(cfg.min_sl_pct)), Decimal(str(cfg.max_sl_pct))
        if sl_distance_pct < min_sl:
            raise _GateFailure(
                "sl_distance", f"SL distance {sl_distance_pct:.3f}% < min {min_sl}%"
            )
        if sl_distance_pct > max_sl:
            raise _GateFailure(
                "sl_distance", f"SL distance {sl_distance_pct:.3f}% > max {max_sl}%"
            )
        passed.append("sl_distance")

        # 10. ATR sanity (flash-wick guard) ----------------------------------------
        atr = to_decimal(features.atr_14_1h, "atr_14_1h")
        min_atr_distance = atr * Decimal(str(cfg.min_sl_atr_multiple))
        if sl_distance < min_atr_distance:
            raise _GateFailure(
                "sl_atr_sanity",
                f"SL distance {sl_distance} < {cfg.min_sl_atr_multiple}x ATR(14) "
                f"({min_atr_distance}) — flash-wick risk",
            )
        warn_atr_distance = atr * Decimal(str(cfg.warn_sl_atr_multiple))
        if sl_distance < warn_atr_distance:
            warnings.append(
                f"SL distance {sl_distance} below {cfg.warn_sl_atr_multiple}x ATR(14)"
            )
        passed.append("sl_atr_sanity")

        # 11. TP side -----------------------------------------------------------------
        tps = [to_decimal(tp, f"take_profit_prices[{i}]")
               for i, tp in enumerate(decision.take_profit_prices)]
        if not tps:
            raise _GateFailure("tp_side", "at least one take-profit price is required")
        for i, tp in enumerate(tps):
            if is_long and tp <= entry_price:
                raise _GateFailure("tp_side", f"long TP[{i}] {tp} must be above entry {entry_price}")
            if not is_long and tp >= entry_price:
                raise _GateFailure("tp_side", f"short TP[{i}] {tp} must be below entry {entry_price}")
        passed.append("tp_side")

        # 12. Minimum R:R (TP1 vs SL) ------------------------------------------------
        tp1_distance = abs(tps[0] - entry_price)
        rr = tp1_distance / sl_distance  # sl_distance > 0 guaranteed by gate 9
        min_rr = Decimal(str(cfg.min_rr_ratio))
        if rr < min_rr:
            raise _GateFailure("min_rr", f"R:R {rr:.3f} < minimum {min_rr}")
        passed.append("min_rr")

        # 13. Daily drawdown ---------------------------------------------------------
        if portfolio.daily_start_equity > 0:
            daily_dd_pct = (
                (portfolio.daily_start_equity - portfolio.equity_usdt)
                / portfolio.daily_start_equity * _HUNDRED
            )
            if daily_dd_pct >= Decimal(str(cfg.daily_loss_limit_pct)):
                await self._engage_drawdown_halt(db, "daily", daily_dd_pct)
                raise _GateFailure(
                    "daily_drawdown",
                    f"daily drawdown {daily_dd_pct:.2f}% >= limit {cfg.daily_loss_limit_pct}%",
                )
        else:
            warnings.append(
                "daily_start_equity is zero — daily drawdown gate bypassed (no equity anchor yet)"
            )
        passed.append("daily_drawdown")

        # 14. Weekly drawdown ----------------------------------------------------------
        if portfolio.weekly_start_equity > 0:
            weekly_dd_pct = (
                (portfolio.weekly_start_equity - portfolio.equity_usdt)
                / portfolio.weekly_start_equity * _HUNDRED
            )
            if weekly_dd_pct >= Decimal(str(cfg.weekly_loss_limit_pct)):
                await self._engage_drawdown_halt(db, "weekly", weekly_dd_pct)
                raise _GateFailure(
                    "weekly_drawdown",
                    f"weekly drawdown {weekly_dd_pct:.2f}% >= limit {cfg.weekly_loss_limit_pct}%",
                )
        else:
            warnings.append(
                "weekly_start_equity is zero — weekly drawdown gate bypassed (no equity anchor yet)"
            )
        passed.append("weekly_drawdown")

        # 15. Post-loss cooldown -----------------------------------------------------------
        since_loss = await self._killswitch.seconds_since_last_loss(db)
        if since_loss is not None and since_loss < cfg.post_loss_cooldown_sec:
            remaining = cfg.post_loss_cooldown_sec - since_loss
            raise _GateFailure(
                "post_loss_cooldown", f"cooling down after loss ({remaining:.0f}s remaining)"
            )
        passed.append("post_loss_cooldown")

        # 16. Consecutive losses --------------------------------------------------------------
        losses = await self._killswitch.consecutive_losses(db)
        if losses >= cfg.max_consecutive_losses:
            raise _GateFailure(
                "consecutive_losses",
                f"{losses} consecutive losses >= cap {cfg.max_consecutive_losses}",
            )
        passed.append("consecutive_losses")

        return entry_price

    async def _engage_drawdown_halt(
        self, db: aiosqlite.Connection, kind: str, drawdown_pct: Decimal,
    ) -> None:
        """Drawdown breaches don't just veto — they halt trading (spec §8)."""
        from datetime import datetime, timedelta

        expires = (
            datetime.now(UTC) + timedelta(hours=24) if kind == "daily" else None
        )
        await self._killswitch.engage(
            db,
            f"{kind} drawdown breach: {drawdown_pct:.2f}%",
            context={"drawdown_pct": float(drawdown_pct)},
            expires_at=expires,
        )

    # ------------------------------------------------------------------
    # Gate 17: sizing
    # ------------------------------------------------------------------

    def _gate_sizing(
        self,
        decision: AiDecision,
        entry_price: Decimal,
        portfolio: PortfolioState,
        precision: PrecisionSpec,
        passed: list[str],
    ) -> SizingResult:
        cfg = self._settings
        try:
            sizing = compute_position_size(
                equity_usdt=portfolio.equity_usdt,
                entry_price=entry_price,
                stop_loss_price=to_decimal(decision.stop_loss_price, "stop_loss_price"),
                risk_per_trade_pct=Decimal(str(cfg.risk_per_trade_pct)),
                max_leverage=cfg.max_leverage,
                max_notional_usdt=Decimal(str(cfg.max_notional_per_symbol_usdt)),
                min_notional_usdt=Decimal(str(cfg.min_notional_usdt)),
                precision=precision,
            )
        except SizingError as exc:
            # Structurally impossible inputs at this point indicate an
            # engine bug — still resolve to a veto (fail safe, not open).
            logger.error("sizing raised on gated inputs (engine bug?): %s", exc)
            raise _GateFailure("sizing", f"sizing error: {exc}") from exc

        if not sizing.ok:
            raise _GateFailure("sizing", sizing.reason or "unsizeable")
        passed.append("sizing")
        return sizing
