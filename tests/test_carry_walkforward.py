"""
Sentinel Trader — Walk-Forward Validation Tests.

These pin the two properties that make walk-forward credible:
  1. selection is **past-only** — a name that only turns positive in the test
     window is never chosen (no look-ahead / survivorship leak);
  2. the bootstrap CI is deterministic and brackets the point estimate.

All synthetic — no network.
"""

from __future__ import annotations

import random

from sentinel.carry.simulator import simulate_static_carry
from sentinel.carry.walkforward import (
    bootstrap_ci,
    measure_oos,
    permutation_test,
    select_basket,
    walk_forward,
)


def test_select_basket_prefers_consistent_positive():
    train = {
        "GOOD": [0.0005] * 20,            # consistent, positive → eligible
        "FLIPPY": [0.0006, -0.0006] * 10,  # mean ~0 → not harvestable
        "NEG": [-0.0004] * 20,           # consistently negative → not harvestable long
        "WEAK": [0.00001] * 20,          # positive but carry below min_carry
    }
    picked = select_basket(train, top_n=6, min_consistency=85.0, min_carry_yr=8.0)
    assert picked == ["GOOD"]


def test_no_lookahead_selection():
    # FUTURE is negative across the whole training window and only pays in the
    # test window. A look-ahead-free selector must NOT pick it.
    history = {
        "STEADY": [0.0005] * 15,
        "FUTURE": [-0.0002] * 10 + [0.0009] * 5,
    }
    res = walk_forward(history, train=10, test=5, step=5,
                       min_consistency=85.0, min_carry_yr=8.0, random_baseline=False)
    assert res.n_windows == 1
    selected = res.windows[0].selected
    assert "STEADY" in selected
    assert "FUTURE" not in selected  # the whole point: chosen on the past, not the future


def test_measure_oos_is_equal_weight():
    test = {"A": [0.0005] * 20, "B": [0.0003] * 20}
    a = simulate_static_carry("A", test["A"])
    b = simulate_static_carry("B", test["B"])
    assert a is not None and b is not None
    net, _ = measure_oos(["A", "B"], test)
    assert net == (a.net_yr + b.net_yr) / 2


def test_measure_oos_empty_basket_earns_nothing():
    assert measure_oos([], {"A": [0.0005] * 10}) == (0.0, 0.0)


def test_walk_forward_window_count_and_positive_oos():
    history = {f"S{i}": [0.0005] * 30 for i in range(4)}
    res = walk_forward(history, train=10, test=5, step=5, random_baseline=False)
    # t ∈ {0,5,10,15} satisfy t+train+test ≤ 30 → 4 non-overlapping OOS windows
    assert res.n_windows == 4
    assert res.mean_oos_net_yr > 0  # consistent positive funding pays out OOS


def test_align_truncates_to_common_recent_length():
    # Different lengths → aligned to the shortest, keeping the most-recent tail.
    history = {"A": [0.0005] * 25, "B": [0.0005] * 15}
    res = walk_forward(history, train=10, test=5, step=5, random_baseline=False)
    # both truncated to 15 → exactly one window
    assert res.n_windows == 1


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    vals = [12.0, 18.0, 9.0, 21.0, 15.0, 11.0, 17.0, 14.0]
    p1, lo1, hi1 = bootstrap_ci(vals, n_resamples=2000, seed=42)
    p2, lo2, hi2 = bootstrap_ci(vals, n_resamples=2000, seed=42)
    assert (p1, lo1, hi1) == (p2, lo2, hi2)   # deterministic for a fixed seed
    assert lo1 <= p1 <= hi1
    assert lo1 > 0  # all-positive sample → CI of the mean stays positive


def test_bootstrap_ci_edge_cases():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([7.5]) == (7.5, 7.5, 7.5)
    # constant sample → zero-width CI at the constant
    assert bootstrap_ci([5.0] * 10, n_resamples=500) == (5.0, 5.0, 5.0)


def test_permutation_detects_a_real_screen_edge():
    # genuine persistent payers + pure-noise names → the screen should beat
    # random selection and the p-value should be significant.
    hist = {f"PAY{i}": [0.0005] * 900 for i in range(4)}
    rng = random.Random(0)
    for i in range(8):
        hist[f"NOISE{i}"] = [rng.choice([0.0006, -0.0006]) for _ in range(900)]
    res = permutation_test(hist, n_permutations=200, train=600, test=120, top_n=4, seed=1)
    assert res.real_mean_oos_yr > res.null_mean_oos_yr
    assert res.p_value < 0.05


def test_permutation_not_significant_when_no_real_edge():
    # all coins identical → consistency selection has no advantage over random,
    # so the test must FAIL to reject (this is what makes it honest).
    hist = {f"S{i}": [0.0004] * 900 for i in range(8)}
    res = permutation_test(hist, n_permutations=200, train=600, test=120, top_n=4, seed=1)
    assert res.p_value > 0.05
