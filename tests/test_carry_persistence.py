"""
Sentinel Trader — Carry Persistence + Scheduling Tests.

Pins book round-trip serialisation (open + closed positions survive a save/
load, so a restart resumes exactly), atomic save behaviour, and the pure
settlement/rebalance scheduling decisions.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sentinel.carry.executor import CarryBook
from sentinel.carry.persistence import book_from_dict, book_to_dict, load_book, save_book
from sentinel.carry.run import seconds_until_next_settlement, should_rebalance


def _book_with_positions():
    b = CarryBook(capital_usdt=50_000.0, slippage_pct=0.05)
    b.open("XMR_USDT", 10_000.0, 300.0, 300.0)
    b.settle_funding({"XMR_USDT": 0.0002}, {"XMR_USDT": 300.0})
    b.open("VELVET_USDT", 10_000.0, 0.4, 0.4)
    b.close_position("VELVET_USDT", 0.4, 0.4, reason="left_basket")  # one open, one closed
    return b


def test_book_roundtrip_preserves_state():
    b = _book_with_positions()
    b2 = book_from_dict(book_to_dict(b))
    assert b2.capital_usdt == b.capital_usdt
    assert b2.realized_net == pytest.approx(b.realized_net)
    assert set(b2.positions) == {"XMR_USDT"}
    assert b2.positions["XMR_USDT"].accrued_funding == pytest.approx(
        b.positions["XMR_USDT"].accrued_funding)
    assert len(b2.closed) == 1 and b2.closed[0].close_reason == "left_basket"


def test_save_then_load_resumes_exactly(tmp_path):
    b = _book_with_positions()
    path = tmp_path / "carry_book.json"
    save_book(b, path)
    loaded = load_book(path)
    assert set(loaded.positions) == {"XMR_USDT"}
    # equity continuity: same marks → same equity after restart
    marks = {"XMR_USDT": 300.0}
    assert loaded.equity(marks, marks) == pytest.approx(b.equity(marks, marks))


def test_load_missing_returns_fresh_book(tmp_path):
    book = load_book(tmp_path / "nope.json", default_capital=12_345.0)
    assert book.capital_usdt == 12_345.0 and not book.positions


# ── Scheduling ────────────────────────────────────────────────────────────


def test_next_settlement_picks_following_8h_boundary():
    now = datetime(2026, 6, 16, 9, 30, tzinfo=UTC)   # between 08:00 and 16:00
    secs = seconds_until_next_settlement(now, buffer_min=5)
    # next is 16:05 → 6h35m = 23700s
    assert secs == pytest.approx(23_700.0)


def test_next_settlement_rolls_to_next_day():
    now = datetime(2026, 6, 16, 23, 0, tzinfo=UTC)    # after 16:05, before next 00:05
    secs = seconds_until_next_settlement(now, buffer_min=5)
    # next is 00:05 tomorrow → 1h5m = 3900s
    assert secs == pytest.approx(3_900.0)


def test_rebalance_due_logic():
    now = datetime(2026, 6, 16, 12, 0, tzinfo=UTC)
    assert should_rebalance(now, None)                                   # never run → due
    assert should_rebalance(now, datetime(2026, 6, 15, 11, 0, tzinfo=UTC))  # >24h → due
    assert not should_rebalance(now, datetime(2026, 6, 16, 6, 0, tzinfo=UTC))  # 6h → not due
