"""
Sentinel Trader — Funding-Carry strategy package.

The 2026-06-16 edge investigation concluded that the only strategy surviving
realistic costs + out-of-sample testing is a *static* delta-neutral funding
carry (long spot + short perp) on the highest-**consistency** positive-funding
liquid alts. This package builds that strategy, starting with the scanner that
curates the basket by funding consistency rather than headline rate.
"""

from sentinel.carry.scanner import (
    FundingStats,
    carry_score,
    compute_funding_stats,
    curate_basket,
)

__all__ = [
    "FundingStats",
    "carry_score",
    "compute_funding_stats",
    "curate_basket",
]
