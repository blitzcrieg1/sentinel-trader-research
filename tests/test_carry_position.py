"""
Sentinel Trader — Delta-Neutral Carry Position Tests.

Pins the accounting core: equal-notional construction, the crucial
delta-neutrality property (price moves cancel across the two legs, leaving
funding minus costs as the return), funding accrual + hysteresis, and the
realised-PnL bookkeeping on unwind.
"""

from __future__ import annotations

import pytest

from sentinel.carry.position import CarryPosition


def _open(notional=1000.0, spot=100.0, perp=100.0):
    return CarryPosition.open_hedge(
        "XMR_USDT", notional, spot, perp,
        spot_fee_pct=0.05, perp_fee_pct=0.02, slippage_pct=0.05,
    )


def test_open_hedge_equal_notional_and_entry_cost():
    p = _open(notional=1000.0, spot=100.0, perp=100.0)
    assert p.spot_qty == 10.0 and p.perp_qty == 10.0
    # entry cost = 2 legs; spot 0.10% + perp 0.07% of 1000 = 1.00 + 0.70
    assert abs(p.accrued_costs - (1.00 + 0.70)) < 1e-9


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        CarryPosition.open_hedge("X", 0.0, 100.0, 100.0)
    with pytest.raises(ValueError):
        CarryPosition.open_hedge("X", 1000.0, -1.0, 100.0)


def test_price_move_is_delta_neutral_up():
    p = _open()
    pnl = p.mark(spot_price=110.0, perp_price=110.0)   # +10% both legs
    assert abs(pnl.spot_pnl + pnl.perp_pnl) < 1e-9     # legs cancel exactly
    assert abs(p.net_delta_usdt(110.0, 110.0)) < 1e-9  # still hedged


def test_price_move_is_delta_neutral_down():
    p = _open()
    pnl = p.mark(spot_price=80.0, perp_price=80.0)     # -20% both legs
    assert abs(pnl.spot_pnl + pnl.perp_pnl) < 1e-9


def test_net_return_is_funding_minus_costs_when_hedged():
    p = _open(notional=1000.0)
    for _ in range(100):                               # 100 settlements
        p.accrue_funding(funding_rate=0.0002, perp_mark=100.0)
    pnl = p.mark(spot_price=130.0, perp_price=130.0)   # big price move — cancels
    # funding = 0.0002 * 10 * 100 * 100 = 20.0; net = funding - entry costs
    assert abs(pnl.funding - 20.0) < 1e-9
    assert abs(pnl.net - (20.0 - p.accrued_costs)) < 1e-9
    assert pnl.net > 0


def test_funding_accrual_and_negative_streak():
    p = _open()
    p.accrue_funding(0.0001, 100.0)
    assert p.neg_funding_streak == 0
    p.accrue_funding(-0.0001, 100.0)
    p.accrue_funding(-0.0001, 100.0)
    assert p.neg_funding_streak == 2
    assert not p.should_unwind(exit_streak=3)
    p.accrue_funding(-0.0001, 100.0)
    assert p.should_unwind(exit_streak=3)
    p.accrue_funding(0.0001, 100.0)                    # one positive resets it
    assert p.neg_funding_streak == 0


def test_close_charges_exit_costs_and_marks_closed():
    p = _open(notional=1000.0)
    p.accrue_funding(0.0002, 100.0)
    cost_before = p.accrued_costs
    pnl = p.close(spot_price=100.0, perp_price=100.0, reason="regime_flip")
    assert p.closed and p.close_reason == "regime_flip"
    assert p.accrued_costs > cost_before               # exit costs added
    assert pnl.net == pytest.approx(p.accrued_funding - p.accrued_costs)
    with pytest.raises(RuntimeError):
        p.close(100.0, 100.0)                          # double-close guarded
