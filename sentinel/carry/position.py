"""
Sentinel Trader — Delta-Neutral Carry Position.

The accounting core of the carry strategy: a single hedge made of a **long
spot** leg and an equal-notional **short perp** leg. Held delta-neutral, the
spot gain offsets the perp loss (and vice-versa), so the economic return is
the funding the short perp collects, minus the costs of establishing and
unwinding the hedge.

Sign convention: ``spot_qty`` and ``perp_qty`` are both stored positive; the
spot is long and the perp is short by construction. All money is USDT.

Pure / no I/O — the paper executor and manager build on this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class CarryPnL:
    """Decomposed mark-to-market of a carry position."""

    spot_pnl: float       # long-spot price PnL
    perp_pnl: float       # short-perp price PnL
    funding: float        # funding collected so far (net of any paid)
    costs: float          # fees + slippage incurred
    net: float            # spot + perp + funding - costs


@dataclass
class CarryPosition:
    """One delta-neutral hedge (long spot + short perp) on a symbol."""

    symbol: str
    spot_qty: float          # long spot units
    perp_qty: float          # short perp contracts (positive)
    spot_entry: float
    perp_entry: float
    open_ts: str = field(default_factory=_utcnow)
    accrued_funding: float = 0.0
    accrued_costs: float = 0.0
    neg_funding_streak: int = 0   # consecutive negative settlements (hysteresis)
    closed: bool = False
    close_ts: str | None = None
    close_reason: str | None = None

    # ------------------------------------------------------------------
    @classmethod
    def open_hedge(
        cls,
        symbol: str,
        notional_usdt: float,
        spot_price: float,
        perp_price: float,
        *,
        spot_fee_pct: float = 0.05,
        perp_fee_pct: float = 0.02,
        slippage_pct: float = 0.05,
        ts: str | None = None,
    ) -> CarryPosition:
        """Establish an equal-notional hedge, charging entry fees + slippage on
        both legs up front."""
        if notional_usdt <= 0 or spot_price <= 0 or perp_price <= 0:
            raise ValueError("notional and prices must be positive")
        spot_qty = notional_usdt / spot_price
        perp_qty = notional_usdt / perp_price
        entry_cost = cls._leg_cost(notional_usdt, spot_fee_pct, slippage_pct) \
            + cls._leg_cost(notional_usdt, perp_fee_pct, slippage_pct)
        return cls(
            symbol=symbol,
            spot_qty=spot_qty,
            perp_qty=perp_qty,
            spot_entry=spot_price,
            perp_entry=perp_price,
            open_ts=ts or _utcnow(),
            accrued_costs=entry_cost,
        )

    @staticmethod
    def _leg_cost(notional: float, fee_pct: float, slippage_pct: float) -> float:
        return notional * (fee_pct + slippage_pct) / 100.0

    # ------------------------------------------------------------------
    def accrue_funding(self, funding_rate: float, perp_mark: float) -> float:
        """Apply one 8h settlement. Short perp *receives* funding when the rate
        is positive (longs pay shorts). Returns the amount applied and updates
        the negative-streak counter used for hysteresis exits."""
        amount = funding_rate * self.perp_qty * perp_mark
        self.accrued_funding += amount
        self.neg_funding_streak = self.neg_funding_streak + 1 if funding_rate < 0 else 0
        return amount

    def should_unwind(self, exit_streak: int = 3) -> bool:
        """Unwind only after a *sustained* negative-funding stretch — a single
        negative print must not churn the book."""
        return self.neg_funding_streak >= exit_streak

    def mark(self, spot_price: float, perp_price: float) -> CarryPnL:
        """Mark-to-market the open position (no close costs applied)."""
        spot_pnl = self.spot_qty * (spot_price - self.spot_entry)
        perp_pnl = self.perp_qty * (self.perp_entry - perp_price)
        net = spot_pnl + perp_pnl + self.accrued_funding - self.accrued_costs
        return CarryPnL(spot_pnl, perp_pnl, self.accrued_funding, self.accrued_costs, net)

    def net_delta_usdt(self, spot_price: float, perp_price: float) -> float:
        """Net dollar delta (long spot − short perp). ~0 when hedged; drifts as
        the two legs' prices diverge, signalling a rebalance."""
        return self.spot_qty * spot_price - self.perp_qty * perp_price

    def close(
        self,
        spot_price: float,
        perp_price: float,
        *,
        spot_fee_pct: float = 0.05,
        perp_fee_pct: float = 0.02,
        slippage_pct: float = 0.05,
        reason: str = "regime_flip",
        ts: str | None = None,
    ) -> CarryPnL:
        """Unwind both legs, charging exit costs, and return the realised PnL."""
        if self.closed:
            raise RuntimeError(f"position {self.symbol} already closed")
        spot_notional = self.spot_qty * spot_price
        perp_notional = self.perp_qty * perp_price
        self.accrued_costs += self._leg_cost(spot_notional, spot_fee_pct, slippage_pct) \
            + self._leg_cost(perp_notional, perp_fee_pct, slippage_pct)
        pnl = self.mark(spot_price, perp_price)
        self.closed = True
        self.close_ts = ts or _utcnow()
        self.close_reason = reason
        return pnl
