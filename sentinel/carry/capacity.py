"""
Sentinel Trader — Funding-Carry Capacity Curve.

Turns the "~$100k ceiling" assertion into a *curve you can inspect*. It models
**net yield vs. notional**, because a yield does not scale with capital — the
carry percentage is the same at $1k or $1M; what grows with size is the **cost
drag** from moving a thin market.

Two things erode the edge as size grows:
  1. **Execution impact** — bigger orders move thin alt spot. Modelled with the
     standard *square-root* law: slippage fraction ≈ k · sqrt(notional / ADV).
  2. **Footprint** — past a few percent of daily volume you can't execute
     cleanly, and (deeper, and *not* modelled here — it needs open-interest /
     elasticity data the scanner doesn't collect) your own demand compresses the
     very funding you're harvesting.

IMPORTANT — this is a **model with assumptions** (the impact coefficient `k`, the
turnover, the participation cap), not measured order-book depth. It is a sanity
check, not a guarantee. Always read the sensitivity band before quoting a number.

The core (`slippage_fraction`, `net_yield_yr`, `build_curve`) is pure and
unit-tested. Only `from_symbol` / `main` touch the network. Plotting is optional
(matplotlib is imported lazily so it is never a hard dependency).

    python -m sentinel.carry.capacity --symbol XMR_USDT -v
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass

from sentinel.carry.scanner import compute_funding_stats, fetch_funding_history, fetch_spot_volume
from sentinel.carry.simulator import simulate_static_carry

logger = logging.getLogger(__name__)

_SETTLEMENTS_PER_YEAR = 3 * 365


@dataclass(frozen=True, slots=True)
class CapacityPoint:
    notional_usdt: float
    participation: float    # notional / daily spot volume (0..1+)
    net_yield_yr: float     # % per year after size-dependent impact


@dataclass(frozen=True, slots=True)
class CapacityCurve:
    symbol: str
    gross_yield_yr: float
    daily_volume_usdt: float
    round_trips_yr: float
    impact_coef: float
    participation_cap: float
    min_yield_yr: float
    points: tuple[CapacityPoint, ...]
    practical_capacity_usdt: float
    binding: str            # "participation" | "yield_floor" | "none"


# ---------------------------------------------------------------------------
# Pure core (no network)
# ---------------------------------------------------------------------------


def slippage_fraction(
    notional_usdt: float, daily_volume_usdt: float, *, impact_coef: float
) -> float:
    """Square-root market impact: cost as a fraction of notional for one fill.
    Grows with the square root of participation (notional / daily volume)."""
    if daily_volume_usdt <= 0:
        return float("inf")
    return impact_coef * math.sqrt(notional_usdt / daily_volume_usdt)


def net_yield_yr(
    gross_yield_yr: float,
    notional_usdt: float,
    daily_volume_usdt: float,
    *,
    impact_coef: float,
    round_trips_yr: float,
    base_round_trip_cost: float = 0.0014,
) -> float:
    """Gross carry minus the annualised cost drag. Per round trip we pay the
    fixed fee plus impact on the thin (spot) leg, twice (open + close); that drag
    recurs ``round_trips_yr`` times a year. At notional→0 this returns the
    small-size net yield, matching the simulator."""
    impact = slippage_fraction(notional_usdt, daily_volume_usdt, impact_coef=impact_coef)
    per_round_trip = base_round_trip_cost + 2.0 * impact
    drag_yr = round_trips_yr * per_round_trip * 100.0
    return gross_yield_yr - drag_yr


def _geomspace(lo: float, hi: float, n: int) -> list[float]:
    if n < 2:
        return [lo]
    r = (hi / lo) ** (1.0 / (n - 1))
    return [lo * r**i for i in range(n)]


def build_curve(
    gross_yield_yr: float,
    daily_volume_usdt: float,
    *,
    symbol: str = "",
    round_trips_yr: float = 5.5,
    impact_coef: float = 0.1,
    base_round_trip_cost: float = 0.0014,
    participation_cap: float = 0.05,
    min_yield_yr: float = 8.0,
    grid_min: float = 1_000.0,
    grid_max: float = 1_000_000.0,
    grid_points: int = 60,
) -> CapacityCurve:
    """Sweep notional from ``grid_min`` to ``grid_max`` and find the practical
    capacity — the smallest notional where *either* participation exceeds the cap
    *or* net yield falls below ``min_yield_yr``, whichever binds first."""
    grid = _geomspace(grid_min, grid_max, grid_points)
    points: list[CapacityPoint] = []
    cap: float | None = None
    binding = "none"
    for n in grid:
        part = n / daily_volume_usdt if daily_volume_usdt > 0 else float("inf")
        ny = net_yield_yr(
            gross_yield_yr, n, daily_volume_usdt,
            impact_coef=impact_coef, round_trips_yr=round_trips_yr,
            base_round_trip_cost=base_round_trip_cost,
        )
        points.append(CapacityPoint(n, part, ny))
        if cap is None and (part > participation_cap or ny < min_yield_yr):
            cap = n
            binding = "participation" if part > participation_cap else "yield_floor"
    if cap is None:
        cap = grid[-1]
    return CapacityCurve(
        symbol=symbol, gross_yield_yr=gross_yield_yr, daily_volume_usdt=daily_volume_usdt,
        round_trips_yr=round_trips_yr, impact_coef=impact_coef,
        participation_cap=participation_cap, min_yield_yr=min_yield_yr,
        points=tuple(points), practical_capacity_usdt=cap, binding=binding,
    )


def sensitivity_band(
    gross_yield_yr: float,
    daily_volume_usdt: float,
    *,
    impact_coef: float = 0.1,
    factor: float = 2.0,
    **kwargs: float,
) -> dict[str, float]:
    """Practical capacity under an optimistic (k/factor) and pessimistic
    (k·factor) impact assumption — a deterministic band, not fake Monte-Carlo.
    Higher impact ⇒ smaller capacity, so pessimistic ≤ mid ≤ optimistic."""
    def cap(k: float) -> float:
        return build_curve(
            gross_yield_yr, daily_volume_usdt, impact_coef=k, **kwargs
        ).practical_capacity_usdt
    return {
        "pessimistic": cap(impact_coef * factor),
        "mid": cap(impact_coef),
        "optimistic": cap(impact_coef / factor),
    }


# ---------------------------------------------------------------------------
# Network + CLI
# ---------------------------------------------------------------------------


def from_symbol(symbol: str, **kwargs: float) -> CapacityCurve | None:
    """Fetch the gross carry, spot volume, and realised turnover for ``symbol``
    and build its capacity curve. Returns ``None`` if data is missing."""
    rates = fetch_funding_history(symbol)
    fs = compute_funding_stats(symbol, rates)
    if fs is None or fs.static_carry_yr <= 0:
        logger.warning("no positive carry for %s", symbol)
        return None
    vol = fetch_spot_volume(symbol)
    if not vol:
        logger.warning("no spot volume for %s — can't size capacity", symbol)
        return None
    # estimate annual turnover from the realised hysteresis cycles over the window
    sim = simulate_static_carry(symbol, rates)
    rt_yr = kwargs.pop("round_trips_yr", None)  # type: ignore[arg-type]
    if rt_yr is None and sim is not None and sim.settlements:
        rt_yr = sim.cycles / (sim.settlements / _SETTLEMENTS_PER_YEAR)
    rt_yr = rt_yr or 5.5
    return build_curve(fs.static_carry_yr, vol, symbol=symbol, round_trips_yr=rt_yr, **kwargs)


def format_report(curve: CapacityCurve, band: dict[str, float]) -> str:
    lines = [
        f"CAPACITY CURVE - {curve.symbol}",
        f"  gross carry:           {curve.gross_yield_yr:6.1f} %/yr",
        f"  daily spot volume:     ${curve.daily_volume_usdt:,.0f}",
        f"  est. turnover:         {curve.round_trips_yr:.1f} round-trips/yr",
        f"  impact coef (k):       {curve.impact_coef:.3f}   "
        f"participation cap: {curve.participation_cap:.0%}",
        "",
        f"  {'notional':>12}  {'participation':>13}  {'net yield':>10}",
    ]
    show = {1_000.0, 10_000.0, 50_000.0, 100_000.0, 250_000.0, 1_000_000.0}
    for p in curve.points:
        if any(abs(p.notional_usdt - s) / s < 0.15 for s in show):
            lines.append(
                f"  ${p.notional_usdt:>11,.0f}  {p.participation:>12.1%}  {p.net_yield_yr:>9.1f}%"
            )
    lines += [
        "",
        f"  practical capacity:    ${curve.practical_capacity_usdt:,.0f}  "
        f"(binds on: {curve.binding})",
        f"  sensitivity band:      ${band['pessimistic']:,.0f}  ..  ${band['optimistic']:,.0f}"
        f"  (mid ${band['mid']:,.0f})",
        "  NOTE: model with assumptions (impact coef, turnover, participation cap),",
        "        not measured order-book depth. A sanity check, not a guarantee.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Funding-carry capacity curve")
    p.add_argument("--symbol", required=True, help="perp symbol, e.g. XMR_USDT")
    p.add_argument("--impact-coef", type=float, default=0.1)
    p.add_argument("--participation-cap", type=float, default=0.05)
    p.add_argument("--min-yield", type=float, default=8.0)
    p.add_argument("--round-trips-yr", type=float, default=None, help="override estimated turnover")
    p.add_argument("--grid-max", type=float, default=1_000_000.0)
    p.add_argument("--json", help="optional path to dump the curve as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s", datefmt="%H:%M:%S",
    )

    kwargs: dict[str, float] = dict(
        impact_coef=args.impact_coef, participation_cap=args.participation_cap,
        min_yield_yr=args.min_yield, grid_max=args.grid_max,
    )
    if args.round_trips_yr is not None:
        kwargs["round_trips_yr"] = args.round_trips_yr

    curve = from_symbol(args.symbol, **kwargs)
    if curve is None:
        print(f"no capacity curve for {args.symbol} (missing carry or volume data)")
        return 1
    band = sensitivity_band(
        curve.gross_yield_yr, curve.daily_volume_usdt,
        impact_coef=args.impact_coef, round_trips_yr=curve.round_trips_yr,
        participation_cap=args.participation_cap, min_yield_yr=args.min_yield,
        grid_max=args.grid_max,
    )
    print(format_report(curve, band))

    if args.json:
        payload = {
            "symbol": curve.symbol,
            "gross_yield_yr": curve.gross_yield_yr,
            "daily_volume_usdt": curve.daily_volume_usdt,
            "round_trips_yr": curve.round_trips_yr,
            "impact_coef": curve.impact_coef,
            "practical_capacity_usdt": curve.practical_capacity_usdt,
            "binding": curve.binding,
            "sensitivity_band": band,
            "points": [
                {"notional_usdt": p.notional_usdt, "participation": p.participation,
                 "net_yield_yr": p.net_yield_yr}
                for p in curve.points
            ],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
