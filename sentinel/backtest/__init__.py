"""
Sentinel Trader — Backtest Harness.

A walk-forward backtest that validates whether the strategy's *underlying
signal* carries edge, independent of the LLM.

The live decision comes from an LLM, which can't be cheaply or honestly
replayed over history (cost + training-data lookahead contamination). So
this harness backtests a **deterministic proxy** of the documented
strategy: it reuses the *real* indicator + regime functions from
``sentinel.data.features`` and encodes the exact per-regime rules from the
system prompt's ``REGIME_GUIDANCE`` (buy pullbacks in bull trends, sell
rallies in bear trends, fade Bollinger extremes in high-vol chop, stand
aside in low-vol chop).

If the proxy shows edge, the feature set carries signal and the LLM is
(hopefully) adding judgement on top. If it's flat/negative, that's itself a
critical finding: either the features are noise, or the LLM is the entire
edge — which you then cannot validate cheaply.

Run with::

    python -m sentinel.backtest --data-dir data/historical
    python -m sentinel.backtest --symbols BTC/USDT,ETH/USDT --rr 1.5
"""

from __future__ import annotations

from sentinel.backtest.engine import BacktestConfig, backtest_symbol, run_backtest
from sentinel.backtest.metrics import PerformanceReport, compute_metrics
from sentinel.backtest.strategy import Signal, generate_signals

__all__ = [
    "BacktestConfig",
    "PerformanceReport",
    "Signal",
    "backtest_symbol",
    "compute_metrics",
    "generate_signals",
    "run_backtest",
]
