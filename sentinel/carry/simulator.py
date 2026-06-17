"""
Sentinel Trader — Static Carry Simulator.

Given a funding-rate history, simulates the realised return of a *static*
delta-neutral hold (long spot + short perp) that harvests funding while it
stays positive and steps aside (flat) during sustained negative-funding
stretches — the strategy that survived the 2026-06-16 investigation.

Crucially this is NOT the per-8h rotation that gets killed by turnover: a
position is opened once and held across many settlements, paying the round-
trip cost only on the rare regime flips. The ``exit_streak`` knob requires N
consecutive negative settlements before unwinding (hysteresis), so a single
negative print doesn't churn the book.

Pure / no network — feed it the rates from ``scanner.fetch_funding_history``.
"""

from __future__ import annotations

from dataclasses import dataclass

_SETTLEMENTS_PER_YEAR = 3 * 365


@dataclass(frozen=True, slots=True)
class CarrySim:
    """Realised result of a static-carry hold over one symbol's history."""

    symbol: str
    settlements: int
    deployed_frac: float    # share of settlements actually holding (0..1)
    cycles: int             # number of open→close round trips
    gross_yr: float         # annualised funding collected while deployed (%)
    net_yr: float           # after round-trip costs (%)
    worst_drawdown: float   # most negative cumulative funding run-up (%)


def simulate_static_carry(
    symbol: str,
    rates: list[float],
    *,
    entry_rate: float = 0.00003,   # open when funding ≥ this (≈ +3.3%/yr)
    exit_streak: int = 3,          # unwind after this many consecutive negatives
    round_trip_cost: float = 0.0014,  # 0.05% spot + 0.02% perp, ×2 legs
) -> CarrySim | None:
    """Walk the funding history with hysteresis. Returns ``None`` if empty."""
    n = len(rates)
    if n == 0:
        return None

    holding = False
    neg_run = 0
    cycles = 0
    deployed = 0
    collected = 0.0          # net cumulative funding (fraction of capital)
    costs = 0.0
    peak = 0.0
    worst_dd = 0.0

    for r in rates:
        if holding:
            collected += r          # short-perp/long-spot earns +funding
            deployed += 1
            neg_run = neg_run + 1 if r < 0 else 0
            if neg_run >= exit_streak:
                costs += round_trip_cost  # cost charged on unwind
                holding = False
                neg_run = 0
        elif r >= entry_rate:
            costs += round_trip_cost      # cost charged on open
            cycles += 1
            holding = True
            deployed += 1
            collected += r
        # drawdown of the net equity curve
        equity = collected - costs
        peak = max(peak, equity)
        worst_dd = min(worst_dd, equity - peak)

    net = collected - costs
    deployed_frac = deployed / n
    # annualise by realised funding cadence over the deployed periods
    gross_yr = (collected / deployed * _SETTLEMENTS_PER_YEAR * 100) if deployed else 0.0
    net_yr = (net / deployed * _SETTLEMENTS_PER_YEAR * 100) if deployed else 0.0
    return CarrySim(
        symbol=symbol,
        settlements=n,
        deployed_frac=deployed_frac,
        cycles=cycles,
        gross_yr=gross_yr,
        net_yr=net_yr,
        worst_drawdown=worst_dd * 100,
    )
