"""
Sentinel Trader — Trade Cost Accounting Tests.

``positions.round_trip_fee`` computes the taker fees actually incurred on a
round trip so the ``trades`` table records a truthful ``net_pnl`` (gross PnL
minus costs) that reconciles with the broker's equity delta. These tests pin
the fee math and the crucial honesty property: a marginally-positive gross
trade can be a *net loss* once fees are charged.
"""

from __future__ import annotations

from decimal import Decimal

from sentinel.core.positions import round_trip_fee

D = Decimal


def test_fee_golden():
    # rate 0.06% on (1000 entry + 1015 exit) notional = 0.0006 * 2015 = 1.209
    fee = round_trip_fee(entry_price=100.0, exit_price=101.5, size=10.0,
                         contract_size=D("1"), fee_pct=0.06)
    assert abs(fee - 1.209) < 1e-9


def test_fee_scales_with_contract_size():
    base = round_trip_fee(100.0, 101.5, 10.0, D("1"), 0.06)
    scaled = round_trip_fee(100.0, 101.5, 10.0, D("10"), 0.06)
    assert abs(scaled - base * 10) < 1e-9


def test_fee_zero_when_rate_zero():
    assert round_trip_fee(100.0, 101.5, 10.0, D("1"), 0.0) == 0.0


def test_fee_symmetric_in_notional():
    # Same total notional, swapped entry/exit → identical fee.
    a = round_trip_fee(100.0, 102.0, 5.0, D("1"), 0.06)
    b = round_trip_fee(102.0, 100.0, 5.0, D("1"), 0.06)
    assert abs(a - b) < 1e-12


def test_marginal_gross_win_is_net_loss():
    """A trade up only a hair gross can lose after round-trip fees — the
    whole reason cost accounting matters for the edge question."""
    entry, exit_, size, cs = 100.0, 100.05, 10.0, D("1")
    gross = (exit_ - entry) * size * float(cs)          # +0.50
    fees = round_trip_fee(entry, exit_, size, cs, 0.06)  # ~1.20
    net = gross - fees
    assert gross > 0
    assert net < 0                                       # fees flip it negative
