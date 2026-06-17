"""
Backtest — walk-forward trade simulation.

Takes the per-bar signal frame from ``strategy.generate_signals`` and
simulates trades bar-by-bar with no lookahead:

- A signal on bar *i* (decided at its close) enters at the **open of bar
  i+1**, adjusted for slippage against the trade direction.
- Stop-loss and take-profit are derived from ATR(14) at entry, clamped to
  the live ``min_sl_pct`` / ``max_sl_pct`` bounds, with TP placed at the
  configured reward:risk multiple (default 1.5, matching ``min_rr_ratio``).
- Each subsequent bar is checked for SL/TP touches using its high/low. When
  both are touched in the same bar, the **stop is assumed first**
  (conservative). A position is held at most ``max_hold_bars`` before a
  market exit at the close.
- One position per symbol at a time (mirrors the ``one_per_symbol`` gate).

Results are expressed in **R-multiples** — realised PnL per unit of risk —
so position sizing reduces to "risk ``risk_per_trade_pct`` of equity per
trade", faithfully mirroring ``risk.sizing`` without exchange precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.backtest.loader import load_candles
from sentinel.backtest.strategy import Signal, StrategyParams, generate_signals
from sentinel.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest parameters; risk/cost defaults are pulled from live Settings."""

    rr_ratio: float = 1.5            # TP distance / SL distance (min_rr_ratio)
    sl_atr_mult: float = 1.0         # SL distance = this × ATR(14)
    min_sl_pct: float = 0.3          # clamp SL distance ≥ this % of price
    max_sl_pct: float = 5.0          # clamp SL distance ≤ this % of price
    max_hold_bars: int = 96          # 96 × 15m = 24h max hold
    fee_pct: float = 0.06            # taker fee per fill (paper_fee_pct)
    slippage_pct: float = 0.05       # slippage against direction (paper_slippage_pct)
    risk_per_trade_pct: float = 0.5  # equity risked per trade at −1R

    @classmethod
    def from_settings(cls, s: Settings) -> BacktestConfig:
        return cls(
            rr_ratio=s.min_rr_ratio,
            min_sl_pct=s.min_sl_pct,
            max_sl_pct=s.max_sl_pct,
            fee_pct=s.paper_fee_pct,
            slippage_pct=s.paper_slippage_pct,
            risk_per_trade_pct=s.risk_per_trade_pct,
        )


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """One simulated round-trip trade."""

    symbol: str
    side: str                # 'long' | 'short'
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    r_multiple: float        # net PnL in units of initial risk (after costs)
    outcome: str             # 'tp' | 'sl' | 'timeout'
    regime: str
    bars_held: int


@dataclass
class SymbolResult:
    """All trades for one symbol plus a quick count."""

    symbol: str
    trades: list[BacktestTrade] = field(default_factory=list)


def _sl_tp(entry: float, atr_val: float, side: int, cfg: BacktestConfig) -> tuple[float, float, float]:
    """Return (sl_price, tp_price, sl_distance) for a long(+1)/short(−1) entry."""
    raw_dist = atr_val * cfg.sl_atr_mult
    lo = entry * cfg.min_sl_pct / 100.0
    hi = entry * cfg.max_sl_pct / 100.0
    sl_dist = min(max(raw_dist, lo), hi)
    tp_dist = sl_dist * cfg.rr_ratio
    if side == Signal.LONG.value:
        return entry - sl_dist, entry + tp_dist, sl_dist
    return entry + sl_dist, entry - tp_dist, sl_dist


def simulate_trade(
    side: int,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    start_idx: int,
    cfg: BacktestConfig,
) -> tuple[float, str, int, float] | None:
    """Walk forward from ``start_idx`` to the first SL/TP touch (or timeout).

    Returns ``(exit_price, outcome, exit_idx, r_multiple)`` where ``outcome``
    is ``'sl' | 'tp' | 'timeout'`` and ``r_multiple`` is net of fees/slippage
    in units of the trade's own initial risk. Returns ``None`` when the SL is
    on the wrong side of entry (zero/negative risk distance).

    Conservative tie-break: if a bar touches both SL and TP, the **stop** is
    assumed to fill first.
    """
    sl_dist = abs(entry_price - sl_price)
    if sl_dist <= 0:
        return None
    risk_frac = sl_dist / entry_price

    fee = cfg.fee_pct / 100.0
    slip = cfg.slippage_pct / 100.0
    n = len(closes)
    exit_idx = min(start_idx + cfg.max_hold_bars, n - 1)
    outcome = "timeout"
    exit_price = float(closes[exit_idx])

    for j in range(start_idx, exit_idx + 1):
        hi, lo = highs[j], lows[j]
        if side == Signal.LONG.value:
            if lo <= sl_price:
                exit_price, outcome, exit_idx = sl_price * (1 - slip), "sl", j
                break
            if hi >= tp_price:
                exit_price, outcome, exit_idx = tp_price, "tp", j
                break
        else:
            if hi >= sl_price:
                exit_price, outcome, exit_idx = sl_price * (1 + slip), "sl", j
                break
            if lo <= tp_price:
                exit_price, outcome, exit_idx = tp_price, "tp", j
                break

    direction = 1.0 if side == Signal.LONG.value else -1.0
    gross = direction * (exit_price - entry_price) / entry_price
    net = gross - 2 * fee
    return float(exit_price), outcome, int(exit_idx), float(net / risk_frac)


def backtest_symbol(
    symbol: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    cfg: BacktestConfig,
    params: StrategyParams | None = None,
    confirm_tf_ms: int = 3_600_000,
) -> SymbolResult:
    """Run the walk-forward simulation for a single symbol.

    ``df_15m`` is the base (decision) frame, ``df_1h`` the confirm frame, and
    ``confirm_tf_ms`` the confirm candle period in ms (default 1h) so the
    pair can be 15m/1h, 1h/4h, etc.
    """
    signals = generate_signals(df_15m, df_1h, params, confirm_tf_ms)
    result = SymbolResult(symbol=symbol)
    if signals.empty:
        return result

    ts = signals["timestamp"].to_numpy()
    opens = signals["open"].to_numpy()
    highs = signals["high"].to_numpy()
    lows = signals["low"].to_numpy()
    closes = signals["close"].to_numpy()
    atrs = signals["atr"].to_numpy()
    sig = signals["signal"].to_numpy()
    regimes = signals["regime"].to_numpy()

    slip = cfg.slippage_pct / 100.0
    n = len(signals)

    i = 0
    while i < n - 1:
        side = int(sig[i])
        if side == Signal.NO_TRADE.value:
            i += 1
            continue

        # Enter at next bar's open, slippage against direction.
        entry_idx = i + 1
        raw_entry = opens[entry_idx]
        entry = raw_entry * (1 + slip) if side == Signal.LONG.value else raw_entry * (1 - slip)
        atr_val = atrs[i]
        if not np.isfinite(atr_val) or atr_val <= 0:
            i += 1
            continue

        sl_price, tp_price, _sl_dist = _sl_tp(entry, atr_val, side, cfg)

        sim = simulate_trade(side, entry, sl_price, tp_price, highs, lows, closes, entry_idx, cfg)
        if sim is None:
            i += 1
            continue
        exit_price, outcome, exit_idx, r_multiple = sim

        result.trades.append(
            BacktestTrade(
                symbol=symbol,
                side="long" if side == Signal.LONG.value else "short",
                entry_ts=int(ts[entry_idx]),
                exit_ts=int(ts[exit_idx]),
                entry_price=float(entry),
                exit_price=float(exit_price),
                r_multiple=float(r_multiple),
                outcome=outcome,
                regime=str(regimes[i]),
                bars_held=exit_idx - entry_idx,
            )
        )

        # One position per symbol: resume scanning after the exit bar.
        i = exit_idx + 1

    return result


def run_backtest(
    data_dir: Path,
    symbols: list[str],
    cfg: BacktestConfig | None = None,
    params: StrategyParams | None = None,
    settings: Settings | None = None,
) -> list[BacktestTrade]:
    """Backtest every symbol and return the pooled trade list (time-sorted)."""
    settings = settings or get_settings()
    cfg = cfg or BacktestConfig.from_settings(settings)

    all_trades: list[BacktestTrade] = []
    for symbol in symbols:
        try:
            df_15m = load_candles(data_dir, symbol, "15m")
            df_1h = load_candles(data_dir, symbol, "1h")
        except Exception as exc:  # noqa: BLE001 — skip a symbol, keep going
            logger.warning("skipping %s: %s", symbol, exc)
            continue
        res = backtest_symbol(symbol, df_15m, df_1h, cfg, params)
        logger.info("%s: %d trades", symbol, len(res.trades))
        all_trades.extend(res.trades)

    all_trades.sort(key=lambda t: t.entry_ts)
    return all_trades
