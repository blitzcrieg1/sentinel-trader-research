"""
Sentinel Trader — Capacity-Curve Tests.

Pin the model's economics: slippage grows with the square root of size, net
yield falls as notional rises, and the practical capacity binds on whichever
constraint (participation cap or yield floor) trips first.

All synthetic — no network.
"""

from __future__ import annotations

import math

from sentinel.carry.capacity import (
    build_curve,
    net_yield_yr,
    sensitivity_band,
    slippage_fraction,
)


def test_slippage_follows_square_root_law():
    s1 = slippage_fraction(10_000, 1_000_000, impact_coef=0.1)
    s4 = slippage_fraction(40_000, 1_000_000, impact_coef=0.1)
    # 4x notional → 2x slippage (sqrt)
    assert math.isclose(s4 / s1, 2.0, rel_tol=1e-9)


def test_slippage_infinite_without_volume():
    assert slippage_fraction(10_000, 0, impact_coef=0.1) == float("inf")


def test_net_yield_decreases_with_size():
    small = net_yield_yr(20.0, 5_000, 1_000_000, impact_coef=0.1, round_trips_yr=5.5)
    big = net_yield_yr(20.0, 500_000, 1_000_000, impact_coef=0.1, round_trips_yr=5.5)
    assert small > big


def test_net_yield_approaches_gross_at_tiny_size():
    # at notional→0 the only drag is the fixed round-trip fee, not impact
    gross = 20.0
    tiny = net_yield_yr(gross, 1e-6, 1_000_000, impact_coef=0.1, round_trips_yr=5.5)
    assert gross - tiny < 1.0  # within ~1pp (just the base fee drag)


def test_capacity_binds_on_participation_when_market_is_deep():
    # huge volume + tiny impact → yield never drops; participation cap is the limit
    curve = build_curve(20.0, daily_volume_usdt=1_000_000_000, impact_coef=1e-6,
                        round_trips_yr=5.5, participation_cap=0.05, min_yield_yr=8.0,
                        grid_max=200_000_000)  # grid must reach 5% of $1B = $50M
    assert curve.binding == "participation"
    # 5% of $1B ≈ $50M
    assert math.isclose(curve.practical_capacity_usdt, 50_000_000, rel_tol=0.2)


def test_capacity_binds_on_yield_when_market_is_thin():
    # thin volume + real impact → net yield collapses well before the 5% cap
    curve = build_curve(20.0, daily_volume_usdt=300_000, impact_coef=0.3,
                        round_trips_yr=8.0, participation_cap=0.05, min_yield_yr=8.0)
    assert curve.binding == "yield_floor"


def test_sensitivity_band_orders_pessimistic_below_optimistic():
    band = sensitivity_band(20.0, 500_000, impact_coef=0.1, round_trips_yr=6.0,
                            participation_cap=0.05, min_yield_yr=8.0)
    # higher impact → smaller capacity
    assert band["pessimistic"] <= band["mid"] <= band["optimistic"]
