"""
Sentinel Trader — Carry Manager Tests.

Pins the pure rebalance logic (open new basket names, close names that left,
never churn a still-good position), the funding-vol risk proxy, and the book's
manual close_position path. The network-touching parts of CarryManager are
exercised separately/manually.
"""

from __future__ import annotations

import pytest

from sentinel.carry.executor import CarryBook
from sentinel.carry.manager import funding_vol, plan_rebalance
from sentinel.carry.scanner import compute_funding_stats


def test_plan_opens_new_and_keeps_existing():
    held = {"XMR_USDT"}
    targets = {"XMR_USDT": 5_000.0, "EVAA_USDT": 5_000.0}
    plan = plan_rebalance(held, targets)
    assert plan.opens == {"EVAA_USDT": 5_000.0}  # only the new name opens
    assert plan.closes == []                     # XMR stays — never churned


def test_plan_closes_names_that_left_basket():
    held = {"XMR_USDT", "OLD_USDT"}
    targets = {"XMR_USDT": 5_000.0}
    plan = plan_rebalance(held, targets)
    assert plan.closes == ["OLD_USDT"]
    assert plan.opens == {}


def test_plan_noop_when_book_matches_basket():
    plan = plan_rebalance({"A", "B"}, {"A": 1.0, "B": 1.0})
    assert plan.is_noop


def test_funding_vol_rewards_consistency():
    perfect = compute_funding_stats("A", [0.0001] * 100)       # 100% consistent
    choppy = compute_funding_stats("B", [0.0001] * 60 + [-0.0001] * 40)  # 60%
    assert funding_vol(perfect) < funding_vol(choppy)
    assert funding_vol(perfect) >= 0.02                        # floor respected


def test_close_position_books_pnl_and_frees_slot():
    b = CarryBook(capital_usdt=100_000.0)
    b.open("XMR_USDT", 10_000.0, 300.0, 300.0)
    b.settle_funding({"XMR_USDT": 0.0002}, {"XMR_USDT": 300.0})
    realized = b.close_position("XMR_USDT", 300.0, 300.0, reason="left_basket")
    assert "XMR_USDT" not in b.positions
    assert b.closed[0].close_reason == "left_basket"
    assert b.realized_net == pytest.approx(realized)
    with pytest.raises(KeyError):
        b.close_position("XMR_USDT", 300.0, 300.0)            # already gone
