"""
Sentinel Trader — Carry Telegram Report Formatter Tests.

Pins the report text: equity + PnL line, deployed/positions summary, per-name
funding, and totals. (The sender itself is network/env-bound and tested
manually.)
"""

from __future__ import annotations

from sentinel.carry.executor import CarryBook
from sentinel.carry.notify import format_carry_report


def test_report_contains_equity_positions_and_funding():
    b = CarryBook(capital_usdt=10_000.0)
    b.open("XMR_USDT", 3_000.0, 300.0, 300.0)
    b.open("VELVET_USDT", 3_000.0, 0.40, 0.40)
    for _ in range(10):
        b.settle_funding({"XMR_USDT": 0.0002, "VELVET_USDT": 0.0001}, {"XMR_USDT": 300.0, "VELVET_USDT": 0.40})

    marks_spot = {"XMR_USDT": 300.0, "VELVET_USDT": 0.40}
    marks_perp = {"XMR_USDT": 300.0, "VELVET_USDT": 0.40}
    text = format_carry_report(b, marks_spot, marks_perp, header="CARRY TEST")

    assert "CARRY TEST" in text
    assert "Equity:" in text
    assert "2 positions" in text
    assert "XMR_USDT" in text and "VELVET_USDT" in text
    assert "Funding accrued:" in text


def test_report_handles_empty_book():
    b = CarryBook(capital_usdt=10_000.0)
    text = format_carry_report(b, {}, {})
    assert "0 positions" in text
    assert "Equity: $10,000.00" in text  # flat book → equity equals capital
