"""
Sentinel Trader — Carry Book (paper portfolio) + risk-parity sizing.

``CarryBook`` is the portfolio manager for the static delta-neutral carry
strategy: it opens hedges from the curated basket, settles funding each 8h,
unwinds on the hysteresis signal, and marks equity. It holds many names at
once — the breadth that, per the Fundamental Law of Active Management
(IR = IC·√breadth), turns a modest per-name carry into a high portfolio
Sharpe. Sizing is inverse-volatility (risk parity) with a per-name cap so no
single token dominates the book.

Pure / no I/O — prices and funding rates are passed in. The live price-fetch
wrapper and the periodic manager loop build on this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinel.carry.position import CarryPosition


def risk_parity_notionals(
    vols: dict[str, float],
    capital_usdt: float,
    *,
    deploy_frac: float = 1.0,
    max_per_name_frac: float = 0.40,
) -> dict[str, float]:
    """Inverse-volatility (risk-parity) notionals across the basket.

    Weight ∝ 1/vol, with each name capped at ``max_per_name_frac`` (excess
    redistributed to uncapped names), then scaled to ``capital * deploy_frac``.
    Names with a non-positive vol are skipped; equal vols → equal weights. If
    too few names exist for the caps to sum to 1, the book prudently
    under-deploys (holds cash) rather than over-concentrating — breadth (many
    names) avoids this.
    """
    inv = {s: 1.0 / v for s, v in vols.items() if v > 0}
    if not inv:
        return {}
    total = sum(inv.values())
    weights = {s: w / total for s, w in inv.items()}

    # Iteratively cap names over the per-name limit, redistributing the freed
    # weight among the still-uncapped names until stable. (A single cap-then-
    # renormalise would just inflate the capped name straight back over the cap.)
    capped: set[str] = set()
    for _ in range(len(weights)):
        violators = [
            s for s in weights
            if s not in capped and weights[s] > max_per_name_frac + 1e-12
        ]
        if not violators:
            break
        for s in violators:
            weights[s] = max_per_name_frac
            capped.add(s)
        remaining = 1.0 - sum(weights[s] for s in capped)
        uncapped = [s for s in weights if s not in capped]
        pool = sum(weights[s] for s in uncapped)
        if pool > 0 and remaining > 0:
            for s in uncapped:
                weights[s] = weights[s] / pool * remaining

    deployable = capital_usdt * deploy_frac
    return {s: deployable * w for s, w in weights.items()}


@dataclass
class CarryBook:
    """Paper portfolio of delta-neutral carry positions."""

    capital_usdt: float
    spot_fee_pct: float = 0.05
    perp_fee_pct: float = 0.02
    slippage_pct: float = 0.05
    positions: dict[str, CarryPosition] = field(default_factory=dict)
    closed: list[CarryPosition] = field(default_factory=list)
    realized_net: float = 0.0

    # ------------------------------------------------------------------
    def open(
        self, symbol: str, notional_usdt: float, spot_price: float, perp_price: float,
        *, ts: str | None = None,
    ) -> CarryPosition:
        """Open a hedge for ``symbol`` (idempotency guarded: one per symbol)."""
        if symbol in self.positions:
            raise ValueError(f"position already open on {symbol}")
        pos = CarryPosition.open_hedge(
            symbol, notional_usdt, spot_price, perp_price,
            spot_fee_pct=self.spot_fee_pct, perp_fee_pct=self.perp_fee_pct,
            slippage_pct=self.slippage_pct, ts=ts,
        )
        self.positions[symbol] = pos
        return pos

    def settle_funding(self, rates: dict[str, float], perp_marks: dict[str, float]) -> float:
        """Apply one 8h settlement to every open position with data. Returns the
        total funding accrued this settlement (USDT)."""
        total = 0.0
        for sym, pos in self.positions.items():
            if sym in rates and sym in perp_marks:
                total += pos.accrue_funding(rates[sym], perp_marks[sym])
        return total

    def process_exits(
        self, spot_prices: dict[str, float], perp_prices: dict[str, float],
        *, exit_streak: int = 3, ts: str | None = None,
    ) -> list[str]:
        """Close positions flagged for unwind by the hysteresis rule. Returns
        the symbols closed."""
        closed_syms: list[str] = []
        for sym in list(self.positions):
            pos = self.positions[sym]
            if pos.should_unwind(exit_streak) and sym in spot_prices and sym in perp_prices:
                pnl = pos.close(
                    spot_prices[sym], perp_prices[sym],
                    spot_fee_pct=self.spot_fee_pct, perp_fee_pct=self.perp_fee_pct,
                    slippage_pct=self.slippage_pct, reason="regime_flip", ts=ts,
                )
                self.realized_net += pnl.net
                self.closed.append(pos)
                del self.positions[sym]
                closed_syms.append(sym)
        return closed_syms

    def close_position(
        self, symbol: str, spot_price: float, perp_price: float,
        *, reason: str = "left_basket", ts: str | None = None,
    ) -> float:
        """Manually unwind one position (e.g. it dropped out of the basket).
        Returns the realised net PnL."""
        pos = self.positions.get(symbol)
        if pos is None:
            raise KeyError(f"no open position on {symbol}")
        pnl = pos.close(
            spot_price, perp_price,
            spot_fee_pct=self.spot_fee_pct, perp_fee_pct=self.perp_fee_pct,
            slippage_pct=self.slippage_pct, reason=reason, ts=ts,
        )
        self.realized_net += pnl.net
        self.closed.append(pos)
        del self.positions[symbol]
        return pnl.net

    # ------------------------------------------------------------------
    def deployed_notional(self) -> float:
        """Sum of spot-leg notionals currently held (the deployed capital)."""
        return sum(p.spot_qty * p.spot_entry for p in self.positions.values())

    def total_funding_accrued(self) -> float:
        """Funding collected across open + closed positions."""
        return (
            sum(p.accrued_funding for p in self.positions.values())
            + sum(p.accrued_funding for p in self.closed)
        )

    def equity(self, spot_prices: dict[str, float], perp_prices: dict[str, float]) -> float:
        """Capital + realised net + unrealised net of all open positions."""
        eq = self.capital_usdt + self.realized_net
        for sym, pos in self.positions.items():
            if sym in spot_prices and sym in perp_prices:
                eq += pos.mark(spot_prices[sym], perp_prices[sym]).net
        return eq
