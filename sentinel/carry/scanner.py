"""
Sentinel Trader — Funding-Carry Scanner.

Ranks MEXC perpetuals by *harvestable* funding carry. The ranking favours
**consistency** (persistent same-sign funding) over headline rate, because the
edge that survived testing is a static delta-neutral hold (long spot + short
perp) — and a static hold only works while funding stays on one side. A 100%-
positive name at +13%/yr beats a 60%-positive name at +40%/yr, because the
latter's negative stretches force costly unwinds.

The scoring/ranking layer (top of file) is pure and unit-tested. The network
fetch layer (bottom) hits MEXC's public contract API and is kept sync/urllib,
matching the analytical ``backtest`` package rather than the async hot path.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)

# Polite pacing so a full-universe scan doesn't trip MEXC's 510 rate limit.
_PAGE_DELAY_SEC = 0.15
_SYMBOL_DELAY_SEC = 0.25

# 8h funding → 3 settlements/day.
_SETTLEMENTS_PER_YEAR = 3 * 365
_TICKER_URL = "https://contract.mexc.com/api/v1/contract/ticker"
_FUNDING_HIST_URL = "https://contract.mexc.com/api/v1/contract/funding_rate/history"
_SPOT_TICKER_URL = "https://api.mexc.com/api/v3/ticker/24hr"


@dataclass(frozen=True, slots=True)
class FundingStats:
    """Funding-carry summary for one symbol over its available history."""

    symbol: str
    n: int                  # number of settlements observed
    mean_8h: float          # mean funding rate per 8h (fraction, e.g. 0.0001)
    pct_positive: float     # share of settlements with positive funding (0..100)
    static_carry_yr: float  # annualised carry if held statically (% / year)
    consistency: float      # how one-sided funding is (0..100; 100 = never flips)
    spot_vol_usdt: float | None = None  # 24h spot quote volume (None = unknown/not listed)

    @property
    def harvestable(self) -> bool:
        """Positive-funding (long spot / short perp — no spot borrow needed)
        and persistent enough for a static hold."""
        return self.mean_8h > 0 and self.consistency >= 80.0

    def tradeable(self, min_spot_vol_usdt: float) -> bool:
        """Harvestable AND the spot leg is liquid enough to hedge. A name with
        great funding but a $0.1M spot market (e.g. BR) can't be executed
        delta-neutral and must not enter the basket."""
        return (
            self.harvestable
            and self.spot_vol_usdt is not None
            and self.spot_vol_usdt >= min_spot_vol_usdt
        )


def compute_funding_stats(symbol: str, rates: list[float]) -> FundingStats | None:
    """Summarise a funding-rate history. Pure — no network. Returns ``None``
    for an empty history."""
    n = len(rates)
    if n == 0:
        return None
    mean = sum(rates) / n
    pct_pos = sum(1 for r in rates if r > 0) / n * 100.0
    static = mean * _SETTLEMENTS_PER_YEAR * 100.0
    consistency = max(pct_pos, 100.0 - pct_pos)
    return FundingStats(symbol, n, mean, pct_pos, static, consistency)


def carry_score(s: FundingStats) -> float:
    """Rank score for the static-carry basket: annual carry weighted by the
    *square* of consistency (so persistence dominates), positive funding only.
    Non-positive funding scores -1 (not harvestable without spot borrow)."""
    if s.mean_8h <= 0:
        return -1.0
    return s.static_carry_yr * (s.consistency / 100.0) ** 2


def curate_basket(
    stats: list[FundingStats],
    *,
    min_consistency: float = 85.0,
    min_carry_yr: float = 8.0,
    min_spot_vol_usdt: float | None = None,
    top_n: int = 6,
) -> list[FundingStats]:
    """Filter to harvestable, persistent, worthwhile names and rank by score.

    When ``min_spot_vol_usdt`` is given, also require a spot market that liquid
    (the hedge's long leg) — a name with great funding but no tradeable spot is
    excluded.
    """
    def eligible(s: FundingStats) -> bool:
        if not (s.mean_8h > 0 and s.consistency >= min_consistency
                and s.static_carry_yr >= min_carry_yr):
            return False
        if min_spot_vol_usdt is not None:
            return s.spot_vol_usdt is not None and s.spot_vol_usdt >= min_spot_vol_usdt
        return True

    return sorted((s for s in stats if eligible(s)), key=carry_score, reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Network fetch layer (sync urllib — analytical tool, not the trading hot path)
# ---------------------------------------------------------------------------


def _get(url: str, timeout: float = 25.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "sentinel-carry/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed host
        return json.load(resp)


def fetch_liquid_universe(min_vol_usdt: float = 3_000_000.0) -> list[str]:
    """All USDT perps whose 24h quote volume clears ``min_vol_usdt``."""
    data = _get(_TICKER_URL).get("data", [])
    out: list[str] = []
    for row in data:
        sym = row.get("symbol", "")
        if not sym.endswith("_USDT"):
            continue
        try:
            vol = float(row.get("amount24", 0))
        except (TypeError, ValueError):
            continue
        if vol >= min_vol_usdt:
            out.append(sym)
    return out


def fetch_spot_volume(symbol: str) -> float | None:
    """24h spot quote volume (USDT) for the perp's spot counterpart, or ``None``
    if the token has no spot listing. ``symbol`` is the perp form ``XMR_USDT``;
    the spot symbol drops the underscore (``XMRUSDT``)."""
    spot_sym = symbol.replace("_", "")
    try:
        data = _get(f"{_SPOT_TICKER_URL}?symbol={spot_sym}")
    except Exception:  # noqa: BLE001 — no spot listing / soft failure
        return None
    try:
        return float(data.get("quoteVolume", 0))
    except (TypeError, ValueError):
        return None


def fetch_funding_history(symbol: str, max_pages: int = 12, page_size: int = 100) -> list[float]:
    """Paginated funding-rate history (most recent first), oldest-truncated at
    ``max_pages``. Best-effort: returns what it got, never raises."""
    rates: list[float] = []
    for page in range(1, max_pages + 1):
        url = f"{_FUNDING_HIST_URL}?symbol={symbol}&page_num={page}&page_size={page_size}"
        try:
            data = _get(url)
        except Exception:  # noqa: BLE001 — soft-fail a page, keep what we have
            break
        if not data.get("success"):
            break
        res = data.get("data", {}).get("resultList", [])
        if not res:
            break
        for x in res:
            try:
                rates.append(float(x["fundingRate"]))
            except (KeyError, TypeError, ValueError):
                continue
        if len(res) < page_size:
            break
        time.sleep(_PAGE_DELAY_SEC)
    return rates


def scan(
    min_vol_usdt: float = 3_000_000.0,
    min_history: int = 200,
    fetch_spot: bool = True,
) -> list[FundingStats]:
    """Fetch the liquid universe and compute funding stats for each name with
    enough history. For harvestable names (positive, persistent) also fetch
    spot volume so the basket can be filtered to what's actually hedgeable.
    Network-bound; returns stats sorted by carry score."""
    universe = fetch_liquid_universe(min_vol_usdt)
    logger.info("carry scan: %d liquid perps in universe", len(universe))
    stats: list[FundingStats] = []
    for i, sym in enumerate(universe, 1):
        rates = fetch_funding_history(sym)
        if len(rates) >= min_history:
            fs = compute_funding_stats(sym, rates)
            if fs is not None:
                if fetch_spot and fs.harvestable:
                    fs = replace(fs, spot_vol_usdt=fetch_spot_volume(sym))
                stats.append(fs)
        if i % 20 == 0:
            logger.info("carry scan: %d/%d processed, %d with history", i, len(universe), len(stats))
        time.sleep(_SYMBOL_DELAY_SEC)
    logger.info("carry scan: complete — %d names analysed", len(stats))
    return sorted(stats, key=carry_score, reverse=True)
