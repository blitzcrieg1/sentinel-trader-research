"""
Sentinel Trader — Risk Engine Gate Tests.

The engine is the fail-closed wall between an AI suggestion and real orders.
These tests drive ``RiskEngine.evaluate`` through its gates with lightweight
stubs (the engine reads only a few fields from features/portfolio, and the
kill switch is faked), verifying:

- The **approval path** passes every gate and returns a sizing.
- Each **capital-protecting veto** fires for the right reason: kill switch,
  confidence floor, SL on the wrong side, SL distance bounds, min R:R,
  position caps, post-loss cooldown, consecutive-loss halt, and drawdown
  breaches (which must also engage the kill switch).

``evaluate`` is async; tests drive it with ``asyncio.run`` so no
pytest-asyncio plugin is required. The kill switch fake ignores ``db``, so a
``None`` connection is passed throughout.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from sentinel.ai.contract import AiDecision, EntrySpec
from sentinel.config import Settings
from sentinel.data.market import PrecisionSpec
from sentinel.risk.engine import RiskEngine

D = Decimal


# ── Fixtures / builders ───────────────────────────────────────────────────


def _settings(**over) -> Settings:
    base = dict(
        _env_file=None,
        mexc_api_key="x", mexc_secret_key="x",
        telegram_admin_chat_id=1, gemini_api_key="dummy",
        max_notional_per_symbol_usdt=1_000_000.0,  # don't let the notional cap bind
    )
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


class _FakeKill:
    """Stand-in kill switch; all methods ignore the db argument."""

    def __init__(self, halted=False, since_loss=None, consec=0):
        self._halted = halted
        self._since = since_loss
        self._consec = consec
        self.engaged: list[str] = []

    async def status(self, db):
        return SimpleNamespace(halted=self._halted, reason="halted-for-test")

    async def seconds_since_last_loss(self, db):
        return self._since

    async def consecutive_losses(self, db):
        return self._consec

    async def engage(self, db, reason, context=None, expires_at=None):
        self.engaged.append(reason)


def _features(symbol="BTC/USDT", price=100.0, atr=1.0, rsi_1h=50.0):
    return SimpleNamespace(
        symbol=symbol, current_price=price, atr_14_1h=atr,
        timeframes={"1h": SimpleNamespace(rsi_14=rsi_1h)},
    )


def _portfolio(equity="10000", positions=None, daily="10000", weekly="10000"):
    return SimpleNamespace(
        equity_usdt=D(equity),
        open_positions=positions or [],
        daily_start_equity=D(daily),
        weekly_start_equity=D(weekly),
    )


def _precision():
    return PrecisionSpec(
        symbol="BTC/USDT", amount_step=D("0.001"), price_step=D("0.01"),
        min_amount=None, min_notional=None, contract_size=D("1"),
    )


def _decision(
    *, symbol="BTC/USDT", decision="long", confidence=0.80, alignment="mixed",
    entry_type="market", limit_price=None, sl=99.0, tps=(102.5,),
) -> AiDecision:
    return AiDecision(
        schema_version="1.0", symbol=symbol, decision=decision, confidence=confidence,
        timeframe_alignment=alignment,
        entry=EntrySpec(type=entry_type, limit_price=limit_price),
        stop_loss_price=sl, take_profit_prices=list(tps),
        invalidation="x", rationale="x", risk_flags=[],
    )


def _evaluate(engine, decision, features, portfolio, precision=None):
    return asyncio.run(engine.evaluate(
        None, decision, features, portfolio, precision or _precision(),
    ))


def _engine(settings=None, kill=None) -> RiskEngine:
    return RiskEngine(settings or _settings(), kill or _FakeKill())


# ── Approval path ─────────────────────────────────────────────────────────


def test_clean_long_is_approved():
    v = _evaluate(_engine(), _decision(), _features(), _portfolio())
    assert v.approved, v.veto_reason
    assert v.sizing is not None and v.sizing.contracts > 0
    assert "min_rr" in v.gates_passed
    assert "sizing" in v.gates_passed


def test_clean_short_is_approved():
    # Short: SL above entry, TP below. RR = 2.5/1.0.
    d = _decision(decision="short", sl=101.0, tps=(97.5,))
    v = _evaluate(_engine(), d, _features(), _portfolio())
    assert v.approved, v.veto_reason


# ── Capital-protecting vetoes ─────────────────────────────────────────────


def test_kill_switch_vetoes_first():
    v = _evaluate(_engine(kill=_FakeKill(halted=True)), _decision(), _features(), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("kill_switch",)


def test_low_confidence_vetoed():
    v = _evaluate(_engine(), _decision(confidence=0.10), _features(), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("confidence",)


def test_long_sl_on_wrong_side_vetoed():
    # Long SL above entry (100) — must be rejected by the sl_side gate.
    v = _evaluate(_engine(), _decision(sl=101.0, tps=(105.0,)), _features(), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("sl_side",)


def test_sl_distance_too_wide_vetoed():
    # SL 90 → 10% distance, above the 5% max.
    v = _evaluate(_engine(), _decision(sl=90.0, tps=(130.0,)), _features(), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("sl_distance",)


def test_sl_distance_too_tight_vetoed():
    # SL 99.9 → 0.1% distance, below the 0.3% min.
    v = _evaluate(_engine(), _decision(sl=99.9, tps=(100.2,)), _features(), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("sl_distance",)


def test_min_rr_vetoed():
    # SL 99 (risk 1.0), TP 100.5 (reward 0.5) → RR 0.5 < 1.5.
    v = _evaluate(_engine(), _decision(sl=99.0, tps=(100.5,)), _features(), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("min_rr",)


def test_max_positions_vetoed():
    s = _settings(max_concurrent_positions=2)
    positions = [
        SimpleNamespace(symbol="ETH/USDT", side="long", contracts=D("1")),
        SimpleNamespace(symbol="SOL/USDT", side="long", contracts=D("1")),
    ]
    v = _evaluate(_engine(s), _decision(), _features(), _portfolio(positions=positions))
    assert not v.approved
    assert v.gates_failed == ("max_positions",)


def test_one_per_symbol_vetoed():
    positions = [SimpleNamespace(symbol="BTC/USDT", side="long", contracts=D("1"))]
    v = _evaluate(_engine(), _decision(symbol="BTC/USDT"), _features(), _portfolio(positions=positions))
    assert not v.approved
    assert v.gates_failed == ("one_per_symbol",)


def test_post_loss_cooldown_vetoed():
    s = _settings(post_loss_cooldown_sec=1800)
    kill = _FakeKill(since_loss=60)  # 60s since loss < 1800s cooldown
    v = _evaluate(_engine(s, kill), _decision(), _features(), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("post_loss_cooldown",)


def test_consecutive_losses_halts():
    s = _settings(max_consecutive_losses=3)
    kill = _FakeKill(consec=3)
    v = _evaluate(_engine(s, kill), _decision(), _features(), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("consecutive_losses",)


def test_daily_drawdown_vetoes_and_engages_killswitch():
    s = _settings(daily_loss_limit_pct=2.0)
    kill = _FakeKill()
    # equity down 3% from the daily anchor → breaches the 2% limit.
    pf = _portfolio(equity="9700", daily="10000", weekly="10000")
    v = _evaluate(_engine(s, kill), _decision(), _features(), pf)
    assert not v.approved
    assert v.gates_failed == ("daily_drawdown",)
    assert kill.engaged and "daily drawdown" in kill.engaged[0]


def test_conflicting_timeframes_need_high_confidence():
    # conflicting alignment with confidence 0.60 < 0.75 required → veto.
    v = _evaluate(_engine(), _decision(alignment="conflicting", confidence=0.60),
                  _features(), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("timeframe_conflict",)


def test_conflicting_timeframes_pass_with_high_confidence():
    v = _evaluate(_engine(), _decision(alignment="conflicting", confidence=0.80),
                  _features(), _portfolio())
    assert v.approved, v.veto_reason


def test_overbought_long_vetoed():
    # 1h RSI 75 > 70 threshold — should block a long entry.
    v = _evaluate(_engine(), _decision(decision="long"), _features(rsi_1h=75.0), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("rsi_extreme",)


def test_oversold_short_vetoed():
    # 1h RSI 25 < 30 threshold — should block a short entry.
    d = _decision(decision="short", sl=101.0, tps=(98.5,))
    v = _evaluate(_engine(), d, _features(rsi_1h=25.0), _portfolio())
    assert not v.approved
    assert v.gates_failed == ("rsi_extreme",)


def test_neutral_rsi_passes():
    # RSI 55 — neither overbought nor oversold, should approve normally.
    v = _evaluate(_engine(), _decision(), _features(rsi_1h=55.0), _portfolio())
    assert v.approved, v.veto_reason
