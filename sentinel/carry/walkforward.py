"""
Sentinel Trader — Walk-Forward Validation for the Funding-Carry Edge.

This answers the sharpest critique of a consistency-screened basket:
*survivorship / look-ahead bias*. A naive backtest picks the basket using the
**full** history and then "tests" on that same history — which is guaranteed to
look good and proves nothing. This module does it honestly:

  * at each rebalance the basket is chosen using **only past** funding data
    (the training window);
  * realised funding is then measured on the **next** window the selector never
    saw (the out-of-sample test window);
  * per-window OOS yields are **bootstrapped** into a confidence interval and
    compared against a **random-selection baseline**, so we can say whether the
    consistency screen actually adds value or just got lucky.

If the edge survives this, it isn't an artefact of picking winners in hindsight.

The core (`select_basket`, `measure_oos`, `walk_forward`, `bootstrap_ci`) is
pure and unit-tested. Only `fetch_histories` / `main` touch the network.

    python -m sentinel.carry.walkforward --scan --train 600 --test 120 --top-k 6

All figures are research estimates from historical funding data, not a live
track record. See docs/METHODOLOGY.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
from dataclasses import dataclass

from sentinel.carry.scanner import (
    carry_score,
    compute_funding_stats,
    fetch_funding_history,
    fetch_liquid_universe,
)
from sentinel.carry.simulator import simulate_static_carry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowResult:
    """One walk-forward step: basket chosen on past data, scored on the future."""

    index: int
    selected: tuple[str, ...]
    oos_net_yr: float        # equal-weight OOS net annualised yield (%) of the basket
    oos_gross_yr: float      # before round-trip costs (%)
    baseline_net_yr: float   # random-basket OOS net yield over the same window (%)


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregate out-of-sample outcome across all walk-forward windows."""

    windows: tuple[WindowResult, ...]
    oos_returns: tuple[float, ...]    # per-window OOS net yields (the bootstrap input)
    baseline_returns: tuple[float, ...]
    mean_oos_net_yr: float
    median_oos_net_yr: float
    mean_baseline_net_yr: float

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def screen_edge_yr(self) -> float:
        """How much the consistency screen beats random selection (pp/yr)."""
        return self.mean_oos_net_yr - self.mean_baseline_net_yr


# ---------------------------------------------------------------------------
# Pure core (no network)
# ---------------------------------------------------------------------------


def select_basket(
    train: dict[str, list[float]],
    *,
    top_n: int = 6,
    min_consistency: float = 85.0,
    min_carry_yr: float = 8.0,
) -> list[str]:
    """Pick the basket from a *training* slice only — the look-ahead-free analogue
    of ``scanner.curate_basket`` (minus the live spot-liquidity filter, which
    needs current market data rather than history)."""
    eligible = []
    for sym, rates in train.items():
        fs = compute_funding_stats(sym, rates)
        if fs is None:
            continue
        if (fs.mean_8h > 0
                and fs.consistency >= min_consistency
                and fs.static_carry_yr >= min_carry_yr):
            eligible.append(fs)
    eligible.sort(key=carry_score, reverse=True)
    return [fs.symbol for fs in eligible[:top_n]]


def measure_oos(
    symbols: list[str],
    test: dict[str, list[float]],
    **sim_kwargs: float,
) -> tuple[float, float]:
    """Equal-weight realised (net, gross) annualised yield of ``symbols`` over a
    *test* slice they were not selected on. Returns (0, 0) for an empty basket
    (under-deployed capital earns nothing — the correct, non-cheating result)."""
    nets, grosses = [], []
    for sym in symbols:
        sim = simulate_static_carry(sym, test.get(sym, []), **sim_kwargs)
        if sim is not None:
            nets.append(sim.net_yr)
            grosses.append(sim.gross_yr)
    if not nets:
        return 0.0, 0.0
    return statistics.fmean(nets), statistics.fmean(grosses)


def _align(history: dict[str, list[float]]) -> tuple[dict[str, list[float]], int]:
    """Truncate every symbol to the common most-recent length so window indices
    line up in time. Input must be chronological **oldest-first** and roughly
    co-terminal (fetched together); we keep the last L settlements of each."""
    lengths = [len(r) for r in history.values() if r]
    if not lengths:
        return {}, 0
    common = min(lengths)
    return {s: r[-common:] for s, r in history.items() if len(r) >= common}, common


def walk_forward(
    history: dict[str, list[float]],
    *,
    train: int = 600,
    test: int = 120,
    step: int | None = None,
    top_n: int = 6,
    min_consistency: float = 85.0,
    min_carry_yr: float = 8.0,
    random_baseline: bool = True,
    seed: int = 0,
    **sim_kwargs: float,
) -> WalkForwardResult:
    """Roll a train→test window across the aligned history. The basket is chosen
    on ``train`` settlements and scored on the following ``test`` settlements it
    never saw; ``step`` defaults to ``test`` (non-overlapping OOS windows)."""
    aligned, length = _align(history)
    step = step or test
    rng = random.Random(seed)
    all_symbols = sorted(aligned)

    windows: list[WindowResult] = []
    oos: list[float] = []
    base: list[float] = []

    t, idx = 0, 0
    while t + train + test <= length:
        train_slice = {s: r[t : t + train] for s, r in aligned.items()}
        test_slice = {s: r[t + train : t + train + test] for s, r in aligned.items()}

        selected = select_basket(
            train_slice, top_n=top_n,
            min_consistency=min_consistency, min_carry_yr=min_carry_yr,
        )
        net, gross = measure_oos(selected, test_slice, **sim_kwargs)

        base_net = 0.0
        if random_baseline and all_symbols:
            pool = list(all_symbols)
            rng.shuffle(pool)
            base_net, _ = measure_oos(pool[:top_n], test_slice, **sim_kwargs)

        windows.append(WindowResult(idx, tuple(selected), net, gross, base_net))
        oos.append(net)
        base.append(base_net)
        t += step
        idx += 1

    return WalkForwardResult(
        windows=tuple(windows),
        oos_returns=tuple(oos),
        baseline_returns=tuple(base),
        mean_oos_net_yr=statistics.fmean(oos) if oos else 0.0,
        median_oos_net_yr=statistics.median(oos) if oos else 0.0,
        mean_baseline_net_yr=statistics.fmean(base) if base else 0.0,
    )


def bootstrap_ci(
    values: list[float] | tuple[float, ...],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``values``. Returns
    (point_estimate, lower, upper). Deterministic for a fixed ``seed``."""
    vals = list(values)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = statistics.fmean(vals)
    if n == 1:
        return point, point, point
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(vals[rng.randrange(n)] for _ in range(n))
        for _ in range(n_resamples)
    )
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return point, lo, hi


@dataclass(frozen=True, slots=True)
class PermutationResult:
    """Empirical significance of the consistency screen vs. random selection."""

    real_mean_oos_yr: float       # OOS yield of the consistency-selected basket
    null_mean_oos_yr: float       # average OOS yield under random selection
    null_p95_oos_yr: float        # 95th percentile of the random-selection null
    p_value: float                # P(random ≥ real); small = screen beats chance
    n_permutations: int


def _random_walk_mean(
    aligned: dict[str, list[float]], length: int, *,
    train: int, test: int, step: int, top_n: int,
    rng: random.Random, sim_kwargs: dict[str, float],
) -> float:
    """One walk-forward pass that selects a *random* basket each window — the
    null model for 'does the consistency screen beat chance selection?'."""
    symbols = sorted(aligned)
    oos: list[float] = []
    t = 0
    while t + train + test <= length:
        test_slice = {s: r[t + train : t + train + test] for s, r in aligned.items()}
        pool = list(symbols)
        rng.shuffle(pool)
        net, _ = measure_oos(pool[:top_n], test_slice, **sim_kwargs)
        oos.append(net)
        t += step
    return statistics.fmean(oos) if oos else 0.0


def permutation_test(
    history: dict[str, list[float]], *,
    n_permutations: int = 1_000, seed: int = 0,
    train: int = 600, test: int = 120, step: int | None = None, top_n: int = 6,
    min_consistency: float = 85.0, min_carry_yr: float = 8.0,
    **sim_kwargs: float,
) -> PermutationResult:
    """Multiple-testing / selection-bias guard. Compares the real
    consistency-selected OOS yield against a null built from many *random* basket
    selections over the same universe and windows, returning an empirical p-value
    P(random ≥ real). A small p means the screen genuinely beats chance.

    Scope (be honest about it): this isolates the value of the *selection rule*
    vs. chance from the *same* universe. It does NOT correct for survivorship in
    the universe itself (only currently-listed names) — that needs point-in-time
    listing data the scanner doesn't collect."""
    aligned, length = _align(history)
    step = step or test
    real = walk_forward(
        history, train=train, test=test, step=step, top_n=top_n,
        min_consistency=min_consistency, min_carry_yr=min_carry_yr,
        random_baseline=False, **sim_kwargs,
    ).mean_oos_net_yr
    rng = random.Random(seed)
    null = [
        _random_walk_mean(aligned, length, train=train, test=test, step=step,
                          top_n=top_n, rng=rng, sim_kwargs=sim_kwargs)
        for _ in range(n_permutations)
    ]
    ge = sum(1 for x in null if x >= real)
    p_value = (1 + ge) / (n_permutations + 1)
    null_sorted = sorted(null)
    p95 = null_sorted[min(int(0.95 * n_permutations), n_permutations - 1)] if null else 0.0
    return PermutationResult(
        real_mean_oos_yr=real,
        null_mean_oos_yr=statistics.fmean(null) if null else 0.0,
        null_p95_oos_yr=p95,
        p_value=p_value,
        n_permutations=n_permutations,
    )


# ---------------------------------------------------------------------------
# Network + CLI (analytical tool — sync urllib via the scanner)
# ---------------------------------------------------------------------------


def fetch_histories(symbols: list[str], *, min_history: int = 200) -> dict[str, list[float]]:
    """Fetch funding history for each symbol and return it **oldest-first**
    (the scanner returns most-recent-first). Drops names with too little data."""
    out: dict[str, list[float]] = {}
    for sym in symbols:
        rates = fetch_funding_history(sym)
        if len(rates) >= min_history:
            out[sym] = rates[::-1]  # reverse → chronological oldest-first
    return out


def format_report(res: WalkForwardResult, ci: tuple[float, float, float]) -> str:
    _point, lo, hi = ci
    lines = [
        "WALK-FORWARD FUNDING-CARRY VALIDATION (out-of-sample)",
        f"  windows (non-overlapping OOS):  {res.n_windows}",
        f"  mean OOS net yield:             {res.mean_oos_net_yr:6.2f} %/yr",
        f"  median OOS net yield:           {res.median_oos_net_yr:6.2f} %/yr",
        f"  95% bootstrap CI of the mean:   [{lo:6.2f}, {hi:6.2f}] %/yr",
        f"  random-basket baseline:         {res.mean_baseline_net_yr:6.2f} %/yr",
        f"  screen edge over random:        {res.screen_edge_yr:6.2f} pp/yr",
    ]
    verdict = (
        "CI excludes 0 and beats random -> the screen adds out-of-sample value."
        if lo > 0 and res.screen_edge_yr > 0
        else "CI includes 0 or fails to beat random -> treat the edge as unproven."
    )
    lines.append(f"  verdict: {verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward validation of the funding-carry edge")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--symbols", help="comma-separated perp symbols, e.g. XMR_USDT,EVAA_USDT")
    g.add_argument("--scan", action="store_true", help="scan the liquid universe instead")
    p.add_argument("--scan-min-vol", type=float, default=3_000_000.0)
    p.add_argument("--max-symbols", type=int, default=40)
    p.add_argument("--train", type=int, default=600, help="training settlements per window")
    p.add_argument("--test", type=int, default=120, help="OOS settlements per window")
    p.add_argument("--step", type=int, default=None, help="advance per window (default: --test)")
    p.add_argument("--top-n", type=int, default=6)
    p.add_argument("--min-consistency", type=float, default=85.0)
    p.add_argument("--min-carry", type=float, default=8.0)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--permutations", type=int, default=0,
                   help="if >0, run the random-selection permutation test (selection-bias guard)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="optional path to dump the full result as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.scan:
        universe = fetch_liquid_universe(args.scan_min_vol)[: args.max_symbols]
    else:
        universe = [s.strip() for s in args.symbols.split(",") if s.strip()]
    logger.info("fetching funding history for %d symbols…", len(universe))
    history = fetch_histories(universe)
    logger.info("usable symbols with enough history: %d", len(history))

    res = walk_forward(
        history, train=args.train, test=args.test, step=args.step, top_n=args.top_n,
        min_consistency=args.min_consistency, min_carry_yr=args.min_carry, seed=args.seed,
    )
    ci = bootstrap_ci(res.oos_returns, n_resamples=args.bootstrap, seed=args.seed)
    print(format_report(res, ci))

    if args.permutations > 0:
        perm = permutation_test(
            history, n_permutations=args.permutations, train=args.train, test=args.test,
            step=args.step, top_n=args.top_n, min_consistency=args.min_consistency,
            min_carry_yr=args.min_carry, seed=args.seed,
        )
        print(
            f"  permutation test ({perm.n_permutations} random baskets): "
            f"real {perm.real_mean_oos_yr:.2f} vs null {perm.null_mean_oos_yr:.2f} %/yr, "
            f"p={perm.p_value:.3f} "
            f"({'beats chance' if perm.p_value < 0.05 else 'NOT significant'})"
        )

    if args.json:
        payload = {
            "n_windows": res.n_windows,
            "mean_oos_net_yr": res.mean_oos_net_yr,
            "median_oos_net_yr": res.median_oos_net_yr,
            "bootstrap_ci_95": {"point": ci[0], "lo": ci[1], "hi": ci[2]},
            "mean_baseline_net_yr": res.mean_baseline_net_yr,
            "screen_edge_yr": res.screen_edge_yr,
            "windows": [
                {"index": w.index, "selected": list(w.selected),
                 "oos_net_yr": w.oos_net_yr, "baseline_net_yr": w.baseline_net_yr}
                for w in res.windows
            ],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
