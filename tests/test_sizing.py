"""
Sentinel Trader — Position Sizing Tests.

``risk.sizing.compute_position_size`` is the code that decides how much money
moves. A silent bug here loses real funds with no error, so it is tested
exhaustively:

1. **Golden cases** — hand-computed inputs where the contract count, risk,
   notional, and leverage are exact by construction.
2. **The core invariant** — post-quantization risk may *never* exceed the
   budget. Asserted as a property over a deterministic grid of inputs.
3. **Quantization** — always floors (risk can only shrink, never grow).
4. **Rejections** — below-minimum notional / amount / quantized-to-zero
   return ``ok=False`` (a veto), not a raise.
5. **Hard errors** — structurally invalid inputs (non-positive equity/price,
   SL == entry) raise ``SizingError`` (they signal upstream bugs).
6. **Long/short symmetry** — sizing depends only on |entry − SL|.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from sentinel.data.market import PrecisionSpec
from sentinel.risk.sizing import (
    SizingError,
    compute_position_size,
    quantize_amount,
)

D = Decimal


def _precision(
    *,
    amount_step: str = "0.001",
    price_step: str = "0.01",
    min_amount: str | None = "0.001",
    min_notional: str | None = "1",
    contract_size: str = "1",
) -> PrecisionSpec:
    return PrecisionSpec(
        symbol="TEST/USDT",
        amount_step=D(amount_step),
        price_step=D(price_step),
        min_amount=D(min_amount) if min_amount is not None else None,
        min_notional=D(min_notional) if min_notional is not None else None,
        contract_size=D(contract_size),
    )


def _size(**overrides):
    kwargs = dict(
        equity_usdt=D("10000"),
        entry_price=D("100"),
        stop_loss_price=D("99"),
        risk_per_trade_pct=D("0.5"),
        max_leverage=3,
        max_notional_usdt=D("1000000"),
        min_notional_usdt=D("1"),
        precision=_precision(),
    )
    kwargs.update(overrides)
    return compute_position_size(**kwargs)


# ── Golden cases ──────────────────────────────────────────────────────────


def test_golden_basic_long():
    """10000 equity, 0.5% risk, 1-wide SL → 50 base units, risk exactly 50."""
    r = _size()
    assert r.ok
    assert r.contracts == D("50")
    assert r.amount_base == D("50")
    assert r.notional_usdt == D("5000")
    assert r.risk_usdt == D("50")
    assert r.actual_risk_pct == D("0.5")
    assert r.leverage == 1                       # notional 5000 < equity 10000
    assert r.margin_usdt == D("5000")


def test_golden_short_symmetry():
    """SL above entry (short) sizes identically — depends only on |entry−SL|."""
    long = _size(entry_price=D("100"), stop_loss_price=D("99"))
    short = _size(entry_price=D("100"), stop_loss_price=D("101"))
    assert short.ok
    assert short.contracts == long.contracts
    assert short.risk_usdt == long.risk_usdt


def test_leverage_and_notional_cap_by_leverage():
    """Tight SL blows up raw notional → capped at equity × max_leverage."""
    r = _size(stop_loss_price=D("99.9"))         # 0.1-wide SL
    assert r.ok
    # notional cap = min(max_notional, equity*lev) = min(1e6, 30000) = 30000
    assert r.notional_usdt == D("30000")
    assert r.contracts == D("300")
    assert r.leverage == 3                        # ceil(30000/10000)
    # Risk is now *below* budget because the notional cap bound the size.
    assert r.actual_risk_pct < D("0.5")
    assert r.risk_usdt == D("30")                 # 300 contracts × 0.1 SL


def test_per_symbol_notional_cap():
    """An explicit per-symbol notional ceiling binds before the leverage cap."""
    r = _size(stop_loss_price=D("99.9"), max_notional_usdt=D("8000"))
    assert r.ok
    assert r.notional_usdt == D("8000")
    assert r.contracts == D("80")


def test_contract_size_scaling():
    """contract_size multiplies base-asset units per contract."""
    r = _size(precision=_precision(contract_size="10"))
    assert r.ok
    # 50 base units / contract_size 10 = 5 contracts
    assert r.contracts == D("5")
    assert r.amount_base == D("50")
    assert r.risk_usdt == D("50")


# ── Quantization floors ───────────────────────────────────────────────────


def test_quantization_floors_never_rounds_up():
    """Coarse amount_step truncates down — risk can only shrink."""
    # raw_amount = 50 base; step 7 → floor(50/7)*7 = 49 contracts (contract_size 1)
    r = _size(precision=_precision(amount_step="7", min_amount=None, min_notional="1"))
    assert r.ok
    assert r.contracts == D("49")
    assert r.actual_risk_pct <= D("0.5")


def test_quantize_amount_helper():
    assert quantize_amount(D("50.9"), D("0.5")) == D("50.5")
    assert quantize_amount(D("9.99"), D("1")) == D("9")
    with pytest.raises(SizingError):
        quantize_amount(D("1"), D("0"))


# ── Rejections (ok=False, not raises) ─────────────────────────────────────


def test_reject_below_min_notional():
    r = _size(equity_usdt=D("10"), min_notional_usdt=D("100"))
    assert not r.ok
    assert r.reason is not None and "notional" in r.reason


def test_reject_quantized_to_zero():
    """Amount step larger than the raw amount floors to zero contracts."""
    r = _size(precision=_precision(amount_step="1000", min_amount=None))
    assert not r.ok
    assert r.reason is not None


def test_reject_below_exchange_min_amount():
    r = _size(precision=_precision(amount_step="0.001", min_amount="100"))
    assert not r.ok
    assert r.reason is not None and "minimum" in r.reason


# ── Hard errors (structural — these are upstream bugs) ────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"equity_usdt": D("0")},
        {"equity_usdt": D("-5")},
        {"entry_price": D("0")},
        {"stop_loss_price": D("0")},
        {"entry_price": D("100"), "stop_loss_price": D("100")},   # zero SL distance
        {"risk_per_trade_pct": D("0")},
        {"max_leverage": 0},
    ],
)
def test_structural_inputs_raise(overrides):
    with pytest.raises(SizingError):
        _size(**overrides)


# ── The core invariant: risk never exceeds budget ─────────────────────────


def test_risk_never_exceeds_budget_grid():
    """Over a deterministic grid, post-quantization risk ≤ budget (×1.0001)."""
    equities = [D("500"), D("10000"), D("250000")]
    entries = [D("0.05"), D("3.2"), D("100"), D("63000")]
    sl_pcts = [D("0.4"), D("1.0"), D("3.5")]
    risks = [D("0.25"), D("0.5"), D("1.0")]
    steps = ["0.0001", "0.001", "1"]

    for eq in equities:
        for entry in entries:
            for sl_pct in sl_pcts:
                sl = entry * (D("1") - sl_pct / D("100"))
                for risk in risks:
                    for step in steps:
                        r = compute_position_size(
                            equity_usdt=eq,
                            entry_price=entry,
                            stop_loss_price=sl,
                            risk_per_trade_pct=risk,
                            max_leverage=5,
                            max_notional_usdt=D("100000000"),
                            min_notional_usdt=D("0"),
                            precision=_precision(
                                amount_step=step, min_amount=None,
                                min_notional=None, contract_size="1",
                            ),
                        )
                        if r.ok:
                            # The headline guarantee: never risk more than the budget.
                            assert r.actual_risk_pct <= risk * D("1.0001"), (
                                f"risk {r.actual_risk_pct} > budget {risk} "
                                f"(eq={eq} entry={entry} sl%={sl_pct} step={step})"
                            )
                            # Margin must fit within equity.
                            assert r.margin_usdt <= eq * D("1.0001")
                            assert r.leverage >= 1
