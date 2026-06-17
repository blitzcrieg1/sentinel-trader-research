"""
Sentinel Trader — Funding-Carry Scanner Tests.

Pins the pure scoring/ranking layer that decides the static-carry basket: the
funding-stats summary, the consistency-weighted score, and basket curation
(positive-funding-only, persistence-gated). No network — the fetch layer is
exercised separately/manually.
"""

from __future__ import annotations

from dataclasses import replace

from sentinel.carry import carry_score, compute_funding_stats, curate_basket
from sentinel.carry.simulator import simulate_static_carry


def _rates(pos_frac: float, mag: float, n: int = 400) -> list[float]:
    """n settlements, ``pos_frac`` of them +mag and the rest -mag."""
    n_pos = round(n * pos_frac)
    return [mag] * n_pos + [-mag] * (n - n_pos)


def test_empty_history_returns_none():
    assert compute_funding_stats("X_USDT", []) is None


def test_all_positive_is_max_consistency():
    s = compute_funding_stats("VELVET_USDT", _rates(1.0, 0.0001))
    assert s is not None
    assert s.pct_positive == 100.0
    assert s.consistency == 100.0
    assert s.mean_8h > 0
    assert s.harvestable


def test_static_carry_annualisation():
    # constant +0.01%/8h → 0.0001 * 3 * 365 * 100 = 10.95%/yr
    s = compute_funding_stats("A_USDT", [0.0001] * 100)
    assert abs(s.static_carry_yr - 10.95) < 1e-6


def test_consistency_is_symmetric_for_negative_names():
    # 90% negative funding → consistency 90 (one-sided), but NOT harvestable
    s = compute_funding_stats("BEAR_USDT", _rates(0.10, 0.0002))
    assert s.consistency == 90.0
    assert s.mean_8h < 0
    assert not s.harvestable          # negative funding needs spot borrow
    assert carry_score(s) == -1.0     # excluded from the long-spot basket


def test_score_rewards_consistency_over_raw_rate():
    # B has higher headline carry but worse consistency; A should win on score.
    a = compute_funding_stats("A_USDT", _rates(1.00, 0.0001))   # 100% pos, +10.95%/yr
    b = compute_funding_stats("B_USDT", _rates(0.70, 0.0004))   # 70% pos, higher mean
    # B's raw carry is larger…
    assert b.static_carry_yr > a.static_carry_yr
    # …but A's perfect consistency wins the score.
    assert carry_score(a) > carry_score(b)


def test_curate_basket_filters_and_ranks():
    stats = [
        compute_funding_stats("GOOD1_USDT", _rates(1.00, 0.00012)),  # +13.1%, 100%
        compute_funding_stats("GOOD2_USDT", _rates(0.95, 0.00018)),  # +19.7%, 95%
        compute_funding_stats("CHOPPY_USDT", _rates(0.60, 0.0003)),  # 60% pos — fails consistency
        compute_funding_stats("TINY_USDT", _rates(1.00, 0.00001)),   # +1.1%/yr — fails min carry
        compute_funding_stats("BEAR_USDT", _rates(0.05, 0.0005)),    # negative — excluded
    ]
    basket = curate_basket(stats, min_consistency=85.0, min_carry_yr=8.0, top_n=6)
    syms = [s.symbol for s in basket]
    assert syms == ["GOOD2_USDT", "GOOD1_USDT"]  # both pass; GOOD2 higher score, ranked first


def test_curate_basket_respects_top_n():
    stats = [compute_funding_stats(f"S{i}_USDT", _rates(1.0, 0.0001 * (i + 1))) for i in range(5)]
    assert len(curate_basket(stats, top_n=3)) == 3


def test_spot_liquidity_filter_excludes_thin_spot():
    # Two great-funding names; one has a deep spot market, one is near-untradeable.
    deep = replace(compute_funding_stats("XMR_USDT", _rates(0.94, 0.00018)), spot_vol_usdt=5_900_000)
    thin = replace(compute_funding_stats("BR_USDT", _rates(1.0, 0.00012)), spot_vol_usdt=100_000)
    # Without a spot floor, both qualify on funding alone…
    assert {s.symbol for s in curate_basket([deep, thin])} == {"XMR_USDT", "BR_USDT"}
    # …with a $2M spot floor, the thin one (BR) is dropped — can't hedge it.
    gated = curate_basket([deep, thin], min_spot_vol_usdt=2_000_000)
    assert [s.symbol for s in gated] == ["XMR_USDT"]


def test_tradeable_requires_known_spot_volume():
    s = compute_funding_stats("NEW_USDT", _rates(1.0, 0.00012))  # spot_vol unknown (None)
    assert s.harvestable
    assert not s.tradeable(min_spot_vol_usdt=2_000_000)          # unknown spot → not tradeable
    assert replace(s, spot_vol_usdt=3_000_000).tradeable(2_000_000)


# ── Static-carry simulator ────────────────────────────────────────────────


def test_sim_empty_returns_none():
    assert simulate_static_carry("X_USDT", []) is None


def test_sim_all_positive_holds_throughout_and_nets_positive():
    r = simulate_static_carry("A_USDT", [0.0001] * 300, entry_rate=0.00003)
    assert r.cycles == 1            # opened once
    assert r.deployed_frac == 1.0  # held the whole history
    assert 0 < r.net_yr < r.gross_yr  # costs shave it but it stays positive


def test_sim_zero_cost_means_net_equals_gross():
    rates = [0.0001] * 300
    cheap = simulate_static_carry("A_USDT", rates, round_trip_cost=0.0)
    pricey = simulate_static_carry("A_USDT", rates, round_trip_cost=0.01)
    assert abs(cheap.net_yr - cheap.gross_yr) < 1e-9
    assert pricey.net_yr < cheap.net_yr


def test_sim_hysteresis_holds_through_isolated_negative():
    rates = [0.0001] * 10 + [-0.0001] + [0.0001] * 10
    r = simulate_static_carry("A_USDT", rates, entry_rate=0.00003, exit_streak=3)
    assert r.cycles == 1  # a single negative print must not churn the book


def test_sim_unwinds_on_sustained_negative_then_reenters():
    rates = [0.0001] * 10 + [-0.0001] * 5 + [0.0001] * 10
    r = simulate_static_carry("A_USDT", rates, entry_rate=0.00003, exit_streak=3)
    assert r.cycles == 2  # unwound after the streak, re-entered when positive again
