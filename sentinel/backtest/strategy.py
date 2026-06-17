"""
Backtest — deterministic proxy strategy.

Encodes the documented strategy from the system prompt's ``REGIME_GUIDANCE``
as pure, vectorised rules over OHLCV, reusing the *real* indicator and
regime functions from ``sentinel.data.features`` so the proxy is a faithful
distillation of the live logic rather than a fresh guess.

Per-regime rules (mirroring REGIME_GUIDANCE):
- **Trending Bull**  → long pullbacks: uptrend structure (EMA20>EMA50, close
  above EMA50) with price pulled back to/under the fast EMA, confirmed by a
  positive 1h MACD histogram. No counter-trend shorts.
- **Trending Bear**  → short rallies: mirror of the above.
- **High Vol Chop**  → fade extremes: long at/under the lower Bollinger band,
  short at/over the upper band (revert toward the mean).
- **Low Vol Chop**   → stand aside (no_trade) — the guidance calls these
  setups noise.

Causality: every indicator here is causal (each value depends only on past +
current bars). The 1h regime/MACD are aligned onto 15m bars by the 1h
candle's **close time** (timestamp + 1h), so a 15m bar never sees a 1h
candle that hasn't closed yet — no lookahead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from sentinel.data.features import (
    ADX_TREND_THRESHOLD,
    HIGH_VOL_ATR_MULTIPLE,
    REGIME_HIGH_VOL_CHOP,
    REGIME_LOW_VOL_CHOP,
    REGIME_TRENDING_BEAR,
    REGIME_TRENDING_BULL,
    REGIME_WINDOW,
    adx,
    atr,
    bollinger,
    ema,
    macd,
    rsi,
)

logger = logging.getLogger(__name__)

_MS_1H = 3_600_000

#: Pullback band: how close to the fast EMA price must be to count as a
#: pullback entry (fraction of price).
PULLBACK_BAND = 0.004


class Signal(int, Enum):
    """Per-bar trade direction."""

    NO_TRADE = 0
    LONG = 1
    SHORT = -1


@dataclass(frozen=True, slots=True)
class StrategyParams:
    """Tunable knobs for the proxy (defaults mirror the live config).

    ``mode`` selects the signal family:
    - ``"trend"``   — the live REGIME_GUIDANCE logic (trend pullbacks + chop fades).
    - ``"meanrev"`` — pure RSI mean reversion: long oversold, short overbought.
      The research-backed family that survives fees on short timeframes. Honours
      ``rsi_oversold`` / ``rsi_overbought`` and the optional Bollinger confluence
      and higher-TF trend filters.
    """

    pullback_band: float = PULLBACK_BAND
    require_macd_confirm: bool = True
    mode: str = "trend"                  # "trend" | "meanrev"
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    require_bb_confluence: bool = False  # meanrev: also require close beyond the BB
    trend_filter: bool = True            # meanrev: don't fade a confirmed higher-TF trend


def regime_series(df_1h: pd.DataFrame) -> pd.Series:
    """Per-bar market regime over a 1h frame, vectorised.

    Replicates ``sentinel.data.features.detect_market_regime`` semantics but
    for every bar rather than just the last: ADX>25 → trend (direction from
    DI±); else compare ATR% to its rolling ``REGIME_WINDOW`` median to split
    high- vs low-vol chop. Bars before warm-up are labelled low-vol chop
    (the most conservative / stand-aside regime).
    """
    high, low, close = df_1h["high"], df_1h["low"], df_1h["close"]
    adx_series, di_plus, di_minus = adx(high, low, close, 14)
    atr_series = atr(high, low, close, 14)
    atr_pct = atr_series / close * 100.0
    atr_pct_median = atr_pct.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW).median()

    trending = adx_series > ADX_TREND_THRESHOLD
    bull = trending & (di_plus >= di_minus)
    bear = trending & (di_plus < di_minus)
    high_vol = (~trending) & (atr_pct > atr_pct_median * HIGH_VOL_ATR_MULTIPLE)

    out = pd.Series(REGIME_LOW_VOL_CHOP, index=df_1h.index, dtype=object)
    out[high_vol] = REGIME_HIGH_VOL_CHOP
    out[bear] = REGIME_TRENDING_BEAR
    out[bull] = REGIME_TRENDING_BULL
    # Bars where ADX/median haven't warmed up stay low-vol chop (stand aside).
    out[adx_series.isna()] = REGIME_LOW_VOL_CHOP
    return out


def _align_confirm_to_base(
    df_base: pd.DataFrame,
    df_confirm: pd.DataFrame,
    regime: pd.Series,
    macd_hist: pd.Series,
    confirm_tf_ms: int,
) -> pd.DataFrame:
    """Map the confirm-frame regime + MACD onto base bars by **close time**.

    A confirm candle at timestamp T closes at ``T + confirm_tf_ms``; its
    values are only known from then on. We key the merge on close-time so
    each base bar sees only completed confirm candles (no lookahead).
    """
    src = pd.DataFrame(
        {
            "close_time": df_confirm["timestamp"] + confirm_tf_ms,
            "regime": regime.values,
            "macd_hist_1h": macd_hist.values,
        }
    ).sort_values("close_time")

    merged = pd.merge_asof(
        df_base[["timestamp"]].sort_values("timestamp"),
        src,
        left_on="timestamp",
        right_on="close_time",
        direction="backward",
    )
    merged.index = df_base.index
    return merged


def generate_signals(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    params: StrategyParams | None = None,
    confirm_tf_ms: int = _MS_1H,
) -> pd.DataFrame:
    """Compute a per-base-bar signal frame for one symbol.

    ``df_15m`` is the *base* (decision) timeframe and ``df_1h`` the *confirm*
    (higher) timeframe; ``confirm_tf_ms`` is the confirm candle period in ms
    (default 1h) so the pair can be 15m/1h, 1h/4h, etc. Names keep the
    15m/1h suffixes for continuity with the default config.

    Returns a DataFrame aligned to ``df_15m`` with columns:
        timestamp, open, high, low, close, atr, signal (Signal int), regime.

    The ``signal`` on bar *i* is a decision made at the **close** of bar *i*;
    the engine enters at the open of bar *i+1*, so the signal must never use
    information beyond bar *i*.
    """
    params = params or StrategyParams()

    close = df_15m["close"]
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    rsi14 = rsi(close, 14)
    atr14 = atr(df_15m["high"], df_15m["low"], close, 14)
    bb_upper, _bb_mid, bb_lower = bollinger(close, length=20, num_std=2.0)

    _line, _sig, macd_hist_1h = macd(df_1h["close"], 12, 26, 9)
    regime = regime_series(df_1h)
    aligned = _align_confirm_to_base(df_15m, df_1h, regime, macd_hist_1h, confirm_tf_ms)

    regime_col = aligned["regime"]
    macd_h = aligned["macd_hist_1h"]

    if params.mode == "meanrev":
        # Pure RSI mean reversion: buy oversold, sell overbought.
        long_mr = rsi14 < params.rsi_oversold
        short_mr = rsi14 > params.rsi_overbought
        if params.require_bb_confluence:
            long_mr = long_mr & (close <= bb_lower)
            short_mr = short_mr & (close >= bb_upper)
        if params.trend_filter:
            # Never fade a confirmed higher-TF trend (don't catch a falling knife
            # in a downtrend, or short into a strong uptrend).
            long_mr = long_mr & (regime_col != REGIME_TRENDING_BEAR)
            short_mr = short_mr & (regime_col != REGIME_TRENDING_BULL)
        signal = np.where(
            long_mr, Signal.LONG.value,
            np.where(short_mr, Signal.SHORT.value, Signal.NO_TRADE.value),
        )
    else:
        uptrend = (ema20 > ema50) & (close > ema50)
        downtrend = (ema20 < ema50) & (close < ema50)
        near_ema20_below = close <= ema20 * (1 + params.pullback_band)
        near_ema20_above = close >= ema20 * (1 - params.pullback_band)

        macd_bull = macd_h > 0 if params.require_macd_confirm else True
        macd_bear = macd_h < 0 if params.require_macd_confirm else True

        long_bull = (regime_col == REGIME_TRENDING_BULL) & uptrend & near_ema20_below & macd_bull
        short_bear = (regime_col == REGIME_TRENDING_BEAR) & downtrend & near_ema20_above & macd_bear

        long_chop = (regime_col == REGIME_HIGH_VOL_CHOP) & (close <= bb_lower)
        short_chop = (regime_col == REGIME_HIGH_VOL_CHOP) & (close >= bb_upper)

        signal = np.where(
            long_bull | long_chop, Signal.LONG.value,
            np.where(short_bear | short_chop, Signal.SHORT.value, Signal.NO_TRADE.value),
        )

    out = pd.DataFrame(
        {
            "timestamp": df_15m["timestamp"].values,
            "open": df_15m["open"].values,
            "high": df_15m["high"].values,
            "low": df_15m["low"].values,
            "close": close.values,
            "atr": atr14.values,
            "signal": signal,
            "regime": regime_col.values,
        }
    )
    # Drop warm-up rows where any indicator is NaN — no decisions there.
    out = out[atr14.notna().values & ema50.notna().values & rsi14.notna().values]
    out = out.reset_index(drop=True)
    return out
