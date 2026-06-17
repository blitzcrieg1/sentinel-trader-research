"""
Sentinel Trader — Carry Manager.

Ties the pieces together: curate the basket (scanner) → size it risk-parity
(executor) → reconcile against the open book (open new names, close names that
left the basket) → settle funding + run hysteresis exits each 8h.

``plan_rebalance`` is pure and unit-tested — it decides *what* to change. The
``CarryManager`` applies those decisions against live prices and persists
nothing itself (the runner owns scheduling + persistence).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sentinel.carry.executor import CarryBook, risk_parity_notionals
from sentinel.carry.prices import fetch_market
from sentinel.carry.scanner import FundingStats, curate_basket

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RebalancePlan:
    """What to change to move the book toward the target basket."""

    opens: dict[str, float] = field(default_factory=dict)  # symbol -> target notional
    closes: list[str] = field(default_factory=list)        # symbols that left the basket

    @property
    def is_noop(self) -> bool:
        return not self.opens and not self.closes


def plan_rebalance(held: set[str], target_notionals: dict[str, float]) -> RebalancePlan:
    """Pure: open basket names not yet held; close held names no longer in the
    basket. Names already held *and* still in the basket are left untouched —
    we never churn a working position (turnover is the strategy's enemy)."""
    opens = {s: n for s, n in target_notionals.items() if s not in held}
    closes = [s for s in held if s not in target_notionals]
    return RebalancePlan(opens=opens, closes=closes)


def funding_vol(stats: FundingStats) -> float:
    """Risk proxy for sizing: how variable a name's funding is. We don't have
    the full series here, so approximate from how one-sided it is — a 100%-
    consistent name is the *lowest* risk. vol ∈ (0, 1]."""
    return max(1.0 - stats.consistency / 100.0, 0.02)  # floor so it never divides by ~0


class CarryManager:
    """Orchestrates one rebalance cycle and one funding settlement."""

    def __init__(
        self,
        book: CarryBook,
        *,
        min_consistency: float = 85.0,
        min_carry_yr: float = 10.0,
        min_spot_vol_usdt: float = 2_000_000.0,
        max_names: int = 8,
        deploy_frac: float = 1.0,
        max_per_name_frac: float = 0.30,
        exit_streak: int = 3,
    ) -> None:
        self.book = book
        self.min_consistency = min_consistency
        self.min_carry_yr = min_carry_yr
        self.min_spot_vol_usdt = min_spot_vol_usdt
        self.max_names = max_names
        self.deploy_frac = deploy_frac
        self.max_per_name_frac = max_per_name_frac
        self.exit_streak = exit_streak

    def target_notionals(self, stats: list[FundingStats]) -> dict[str, float]:
        """Curate the basket and size it risk-parity by funding consistency."""
        basket = curate_basket(
            stats, min_consistency=self.min_consistency, min_carry_yr=self.min_carry_yr,
            min_spot_vol_usdt=self.min_spot_vol_usdt, top_n=self.max_names,
        )
        if not basket:
            return {}
        vols = {s.symbol: funding_vol(s) for s in basket}
        return risk_parity_notionals(
            vols, self.book.capital_usdt,
            deploy_frac=self.deploy_frac, max_per_name_frac=self.max_per_name_frac,
        )

    def rebalance(self, stats: list[FundingStats]) -> RebalancePlan:
        """Compute + apply a rebalance against live prices. Returns the plan."""
        targets = self.target_notionals(stats)
        plan = plan_rebalance(set(self.book.positions), targets)
        if plan.is_noop:
            return plan

        market = fetch_market(list(set(plan.opens) | set(plan.closes)))
        for sym in plan.closes:
            m = market.get(sym)
            if m:
                self.book.close_position(sym, m["spot"], m["perp"], reason="left_basket")
                logger.info("carry: closed %s (left basket)", sym)
        for sym, notional in plan.opens.items():
            m = market.get(sym)
            if m:
                self.book.open(sym, notional, m["spot"], m["perp"])
                logger.info("carry: opened %s notional=%.0f", sym, notional)
        return plan

    def settle(self) -> float:
        """Apply one 8h funding settlement to the book + run hysteresis exits.
        Returns funding collected this settlement."""
        if not self.book.positions:
            return 0.0
        market = fetch_market(list(self.book.positions))
        rates = {s: m["funding"] for s, m in market.items()}
        marks = {s: m["perp"] for s, m in market.items()}
        collected = self.book.settle_funding(rates, marks)
        spot = {s: m["spot"] for s, m in market.items()}
        closed = self.book.process_exits(spot, marks, exit_streak=self.exit_streak)
        if closed:
            logger.info("carry: hysteresis-exited %s", closed)
        return collected
