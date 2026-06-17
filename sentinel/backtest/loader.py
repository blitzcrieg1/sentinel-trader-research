"""
Backtest — historical OHLCV loader.

Reads the JSON candle files written by ``sentinel.data.historical`` (one
directory per symbol, one file per timeframe) into pandas DataFrames.

File schema (per ``data/historical/<SYM>/<tf>.json``)::

    {"symbol": "...", "timeframe": "15m", "candles": [[ts_ms, o, h, l, c, v], ...]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class HistoryError(Exception):
    """Raised when historical data is missing or malformed."""


def _symbol_dirname(symbol: str) -> str:
    """``BTC/USDT`` → ``BTC_USDT`` (matches the exporter's directory naming)."""
    return symbol.replace("/", "_").replace(":USDT", "")


def load_candles(data_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load one symbol/timeframe into a clean, time-sorted OHLCV DataFrame.

    The frame is indexed by row position (not timestamp) but carries an
    int64 ``timestamp`` column in epoch-ms. Duplicate timestamps are
    dropped (keep last) and rows are sorted ascending — the indicator and
    walk-forward code both assume a strictly increasing series.
    """
    path = data_dir / _symbol_dirname(symbol) / f"{timeframe}.json"
    if not path.exists():
        raise HistoryError(f"missing history file: {path}")

    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    candles = payload.get("candles")
    if not candles:
        raise HistoryError(f"no candles in {path}")

    df = pd.DataFrame(candles, columns=_COLUMNS)
    df = df.astype(
        {
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp", ignore_index=True)
    return df


def available_symbols(data_dir: Path) -> list[str]:
    """Symbols present in the data dir, read from the manifest when available."""
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        with manifest.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        syms = payload.get("symbols")
        if syms:
            return list(syms)

    # Fallback: infer from directory names.
    return sorted(
        f"{p.name.replace('_', '/')}"
        for p in data_dir.iterdir()
        if p.is_dir()
    )
