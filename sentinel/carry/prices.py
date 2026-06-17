"""
Sentinel Trader — Carry price/funding fetch layer.

Thin network helpers the manager uses to mark the book and settle funding:
current spot price, current perp price, and the current funding rate, from
MEXC's public APIs. Sync/urllib, best-effort (``None`` on failure) — matching
the scanner's fetch layer.
"""

from __future__ import annotations

from sentinel.carry.scanner import _get  # shared urllib helper

_CONTRACT_TICKER = "https://contract.mexc.com/api/v1/contract/ticker"
_SPOT_TICKER = "https://api.mexc.com/api/v3/ticker/price"


def fetch_perp_quote(symbol: str) -> tuple[float, float] | None:
    """(perp last price, current funding rate) for a ``BTC_USDT`` perp, or
    ``None`` on failure."""
    try:
        d = _get(f"{_CONTRACT_TICKER}?symbol={symbol}").get("data", {})
        return float(d["lastPrice"]), float(d["fundingRate"])
    except Exception:  # noqa: BLE001 — best-effort market data
        return None


def fetch_spot_price(symbol: str) -> float | None:
    """Spot last price for the perp's counterpart (``BTC_USDT`` → ``BTCUSDT``)."""
    spot_sym = symbol.replace("_", "")
    try:
        d = _get(f"{_SPOT_TICKER}?symbol={spot_sym}")
        return float(d["price"])
    except Exception:  # noqa: BLE001 — best-effort market data
        return None


def fetch_market(symbols: list[str]) -> dict[str, dict[str, float]]:
    """Batch fetch spot/perp/funding for several symbols. Skips any name whose
    spot or perp quote is unavailable, so callers get only fully-quotable
    markets. Returns ``{symbol: {"spot", "perp", "funding"}}``."""
    out: dict[str, dict[str, float]] = {}
    for sym in symbols:
        perp = fetch_perp_quote(sym)
        spot = fetch_spot_price(sym)
        if perp is None or spot is None:
            continue
        out[sym] = {"spot": spot, "perp": perp[0], "funding": perp[1]}
    return out
