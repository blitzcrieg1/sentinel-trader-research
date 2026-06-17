"""
Sentinel Trader — Carry Book + Sizing Tests.

Pins the portfolio mechanics: risk-parity sizing (inverse-vol, per-name cap),
opening hedges, funding settlement across the book, hysteresis-driven exits
with realised-PnL booking, and equity = capital + realised + unrealised.
"""

from __future__ import annotations

import pytest

from sentinel.carry.executor import CarryBook, risk_parity_notionals


# ── Risk-parity sizing ────────────────────────────────────────────────────


def test_equal_vol_gives_equal_notionals():
    n = risk_parity_notionals({"A": 0.1, "B": 0.1, "C": 0.1}, 30_000.0)
    assert n == pytest.approx({"A": 10_000.0, "B": 10_000.0, "C": 10_000.0})


def test_lower_vol_gets_more_capital():
    n = risk_parity_notionals({"CALM": 0.05, "WILD": 0.20}, 10_000.0, max_per_name_frac=1.0)
    assert n["CALM"] > n["WILD"]              # inverse-vol: calmer name larger
    assert sum(n.values()) == pytest.approx(10_000.0)


def test_per_name_cap_is_enforced():
    # One ultra-low-vol name would dominate; cap holds it to 40%.
    n = risk_parity_notionals({"BIG": 0.001, "X": 0.1, "Y": 0.1}, 10_000.0, max_per_name_frac=0.40)
    assert n["BIG"] <= 0.40 * 10_000.0 + 1e-6
    assert sum(n.values()) == pytest.approx(10_000.0)


def test_deploy_frac_limits_total():
    # No cap (1.0) so this isolates the deploy_frac scaling.
    n = risk_parity_notionals({"A": 0.1, "B": 0.1}, 10_000.0, deploy_frac=0.5, max_per_name_frac=1.0)
    assert sum(n.values()) == pytest.approx(5_000.0)


def test_cap_under_deploys_when_too_few_names():
    # 2 names + 40% cap can't reach 100% — prudently holds the remaining 20%
    # as cash rather than over-concentrating. Breadth (many names) avoids this.
    n = risk_parity_notionals({"A": 0.1, "B": 0.1}, 10_000.0, max_per_name_frac=0.40)
    assert n["A"] == pytest.approx(4_000.0) and n["B"] == pytest.approx(4_000.0)
    assert sum(n.values()) == pytest.approx(8_000.0)  # 20% undeployed


# ── CarryBook ─────────────────────────────────────────────────────────────


def _book():
    return CarryBook(capital_usdt=100_000.0, spot_fee_pct=0.05, perp_fee_pct=0.02, slippage_pct=0.05)


def test_open_and_no_double_open():
    b = _book()
    b.open("XMR_USDT", 10_000.0, spot_price=300.0, perp_price=300.0)
    assert "XMR_USDT" in b.positions
    assert b.deployed_notional() == pytest.approx(10_000.0)
    with pytest.raises(ValueError):
        b.open("XMR_USDT", 5_000.0, 300.0, 300.0)


def test_funding_settlement_accrues_across_book():
    b = _book()
    b.open("XMR_USDT", 10_000.0, 300.0, 300.0)
    b.open("VELVET_USDT", 10_000.0, 0.40, 0.40)
    got = b.settle_funding(
        rates={"XMR_USDT": 0.0002, "VELVET_USDT": 0.0001},
        perp_marks={"XMR_USDT": 300.0, "VELVET_USDT": 0.40},
    )
    # XMR: 0.0002 * (10000/300) * 300 = 2.0 ; VELVET: 0.0001 * (10000/0.40)*0.40 = 1.0
    assert got == pytest.approx(3.0)
    assert b.total_funding_accrued() == pytest.approx(3.0)


def test_equity_is_capital_plus_funding_when_price_neutral():
    b = _book()
    b.open("XMR_USDT", 10_000.0, 300.0, 300.0)
    entry_cost = b.positions["XMR_USDT"].accrued_costs
    for _ in range(50):
        b.settle_funding({"XMR_USDT": 0.0002}, {"XMR_USDT": 300.0})
    # price moves but both legs move together → price PnL cancels
    eq = b.equity({"XMR_USDT": 360.0}, {"XMR_USDT": 360.0})
    funding = b.total_funding_accrued()         # 0.0002 * 33.33 * 300 * 50 = 100.0
    assert eq == pytest.approx(100_000.0 + funding - entry_cost)


def test_hysteresis_exit_books_realized_and_frees_slot():
    b = _book()
    b.open("BSB_USDT", 10_000.0, 1.0, 1.0)
    b.settle_funding({"BSB_USDT": 0.0002}, {"BSB_USDT": 1.0})   # +funding
    for _ in range(3):                                          # sustained negative
        b.settle_funding({"BSB_USDT": -0.0001}, {"BSB_USDT": 1.0})
    closed = b.process_exits({"BSB_USDT": 1.0}, {"BSB_USDT": 1.0}, exit_streak=3)
    assert closed == ["BSB_USDT"]
    assert "BSB_USDT" not in b.positions       # slot freed for a new name
    assert len(b.closed) == 1
    # realised net = collected funding - all costs (price-neutral here)
    assert b.realized_net == pytest.approx(b.closed[0].mark(1.0, 1.0).net)
