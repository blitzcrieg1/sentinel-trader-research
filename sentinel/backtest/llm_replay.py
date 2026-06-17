"""
Backtest — live-LLM replay.

The mechanical backtest answered "do the indicators carry edge?" (no). This
answers the question that actually decides the project: **does the LLM's
judgement beat a no-edge baseline?**

For a sample of historical decision points it reconstructs the exact feature
packet the live model would have seen (reusing ``data.features``), asks the
*real* model for a decision, then simulates that decision forward through the
same SL/TP engine used by the mechanical backtest. The output is the LLM's
realised expectancy in R-multiples, its selectivity (how often it stands
aside), and the reward:risk geometry it chooses — compared against the
random-walk baseline.

Fidelity caveats (stated honestly):
- Historical OHLCV has **no order book or open-interest**, so OBI and
  derivatives features are fed as neutral/empty. The model sees a thinner
  packet than live. This weakens — but does not invalidate — the signal.
- The model was trained on data spanning this period. Feeding it only an
  indicator vector (not a labelled date) limits look-ahead, but it is a
  known asterisk on any historical LLM replay.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.ai.client import AiClient
from sentinel.backtest.engine import BacktestConfig, simulate_trade
from sentinel.backtest.loader import available_symbols, load_candles
from sentinel.backtest.strategy import Signal
from sentinel.config import Settings, get_settings
from sentinel.data.features import MIN_CANDLES, FeatureError, compute_features
from sentinel.data.market import (
    Candle,
    MarketSnapshot,
    OhlcvSeries,
    OrderBookTop,
    TickerSnapshot,
)

logger = logging.getLogger(__name__)

_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
#: Tail length per timeframe fed to the feature engine (> MIN_CANDLES warm-up).
_TAIL = 260


@dataclass(frozen=True, slots=True)
class LlmReplayResult:
    symbol: str
    decisions: int
    no_trades: int
    errors: int
    gated: int          # passed the model but vetoed by a deterministic geometry gate
    trades: list[dict]  # each: side, r_multiple, outcome, rr, confidence, regime


def _passes_geometry_gates(
    side: int, entry: float, sl: float, tp: float, packet, cfg: BacktestConfig, settings: Settings,
) -> bool:
    """Replicate the live risk engine's *geometry* gates (the ones that depend
    only on the model's SL/TP + features, not on portfolio state): SL distance
    band, ATR sanity, min R:R, and the RSI-extreme gate. Trades the live engine
    would veto must not count toward the model's realised edge.
    """
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return False
    sl_dist_pct = sl_dist / entry * 100.0
    if sl_dist_pct < cfg.min_sl_pct or sl_dist_pct > cfg.max_sl_pct:
        return False
    atr_1h = getattr(packet, "atr_14_1h", None)
    if atr_1h and sl_dist < atr_1h * settings.min_sl_atr_multiple:
        return False
    if abs(tp - entry) / sl_dist < cfg.rr_ratio:
        return False
    tf_1h = packet.timeframes.get("1h") if getattr(packet, "timeframes", None) else None
    if tf_1h is not None:
        rsi_1h = getattr(tf_1h, "rsi_14", None)
        if rsi_1h is not None:
            if side == Signal.LONG.value and rsi_1h > settings.rsi_overbought_threshold:
                return False
            if side == Signal.SHORT.value and rsi_1h < settings.rsi_oversold_threshold:
                return False
    return True


def _candles_from_df(df: pd.DataFrame) -> list[Candle]:
    return [
        Candle(
            timestamp_ms=int(r.timestamp), open=float(r.open), high=float(r.high),
            low=float(r.low), close=float(r.close), volume=float(r.volume),
        )
        for r in df.itertuples(index=False)
    ]


def _neutral_order_book(price: float) -> OrderBookTop:
    """Synthetic balanced book (no historical depth) → OBI ≈ 0, tight spread."""
    bid = price * 0.9999
    ask = price * 1.0001
    qty = 10.0
    return OrderBookTop(
        best_bid=bid, best_ask=ask,
        spread_pct=(ask - bid) / ((ask + bid) / 2) * 100.0,
        bid_depth_usdt=bid * qty, ask_depth_usdt=ask * qty,
        bids=((bid, qty),), asks=((ask, qty),),
    )


def _build_snapshot(
    symbol: str,
    candles: dict[str, list[Candle]],
    cutoff_idx: dict[str, int],
    decision_ts: int,
) -> MarketSnapshot | None:
    """Assemble a point-in-time MarketSnapshot from completed candles only."""
    series: dict[str, OhlcvSeries] = {}
    for tf in ("15m", "1h", "4h"):
        end = cutoff_idx[tf]
        if end < MIN_CANDLES:
            return None
        window = candles[tf][max(0, end - _TAIL): end]
        if len(window) < MIN_CANDLES:
            return None
        series[tf] = OhlcvSeries(symbol=symbol, timeframe=tf, candles=tuple(window))

    price = series["15m"].candles[-1].close
    return MarketSnapshot(
        symbol=symbol,
        swap_symbol=f"{symbol}:USDT",
        fetched_at_ms=decision_ts,
        series=series,
        ticker=TickerSnapshot(
            symbol=symbol, last=price, bid=price * 0.9999, ask=price * 1.0001,
            timestamp_ms=decision_ts,
        ),
        funding_rate_8h=None,
        order_book=_neutral_order_book(price),
        open_interest=None,
        open_interest_24h_ago=None,
    )


async def replay_symbol(
    symbol: str,
    data_dir: Path,
    ai: AiClient,
    cfg: BacktestConfig,
    samples: int,
    settings: Settings,
    apply_gates: bool = True,
) -> LlmReplayResult:
    """Replay the live model over ``samples`` decision points for one symbol.

    When ``apply_gates`` is set, a model decision that would be vetoed by the
    live risk engine's geometry gates (min R:R, RSI extreme, SL distance/ATR)
    is counted as ``gated`` and excluded from the trade tape — so the measured
    expectancy reflects what the live system would actually execute.
    """
    df = {tf: load_candles(data_dir, symbol, tf) for tf in ("15m", "1h", "4h")}
    candles = {tf: _candles_from_df(df[tf]) for tf in ("15m", "1h", "4h")}
    ts = {tf: df[tf]["timestamp"].to_numpy() for tf in ("15m", "1h", "4h")}

    highs = df["15m"]["high"].to_numpy()
    lows = df["15m"]["low"].to_numpy()
    opens = df["15m"]["open"].to_numpy()
    closes = df["15m"]["close"].to_numpy()
    n15 = len(closes)

    # Valid decision range: enough 4h warm-up behind, enough 15m forward ahead.
    # The binding warm-up is 4h needing MIN_CANDLES completed candles.
    first_ok_ts = int(ts["4h"][MIN_CANDLES]) + _MS["4h"]
    lo = int(np.searchsorted(ts["15m"], first_ok_ts))
    hi = n15 - cfg.max_hold_bars - 2
    if hi <= lo:
        return LlmReplayResult(symbol, 0, 0, 0, 0, [])

    idxs = np.linspace(lo, hi, num=min(samples, hi - lo), dtype=int)
    slip = cfg.slippage_pct / 100.0

    decisions = no_trades = errors = gated = 0
    trades: list[dict] = []
    for i in idxs:
        decision_ts = int(ts["15m"][i]) + _MS["15m"]  # bar i's close time
        cutoff = {
            "15m": i + 1,  # candles 0..i inclusive
            "1h": int(np.searchsorted(ts["1h"], decision_ts - _MS["1h"], side="right")),
            "4h": int(np.searchsorted(ts["4h"], decision_ts - _MS["4h"], side="right")),
        }
        snap = _build_snapshot(symbol, candles, cutoff, decision_ts)
        if snap is None:
            continue

        try:
            packet = compute_features(snap)
        except FeatureError as exc:
            logger.debug("features failed %s @ %d: %s", symbol, i, exc)
            continue

        decision, meta = await ai.request_decision(packet.to_dict(), [], [])
        decisions += 1
        if decision is None:
            errors += 1
            continue
        if decision.decision == "no_trade":
            no_trades += 1
            continue

        side = Signal.LONG.value if decision.decision == "long" else Signal.SHORT.value
        raw_entry = float(opens[i + 1])
        entry = raw_entry * (1 + slip) if side == Signal.LONG.value else raw_entry * (1 - slip)
        sl = float(decision.stop_loss_price)
        tp = float(decision.take_profit_prices[0])

        if apply_gates and not _passes_geometry_gates(side, entry, sl, tp, packet, cfg, settings):
            gated += 1
            continue

        sim = simulate_trade(side, entry, sl, tp, highs, lows, closes, i + 1, cfg)
        if sim is None:  # SL on wrong side of entry — model produced a bad geometry
            errors += 1
            continue
        _exit_price, outcome, _exit_idx, r_multiple = sim
        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0.0
        sl_dist_pct = abs(entry - sl) / entry * 100.0
        tf_1h = packet.timeframes.get("1h") if getattr(packet, "timeframes", None) else None
        rsi_1h = getattr(tf_1h, "rsi_14", None) if tf_1h is not None else None
        trades.append({
            "symbol": symbol,
            "side": decision.decision,
            "r_multiple": r_multiple,
            "outcome": outcome,
            "rr": rr,
            "sl_dist_pct": sl_dist_pct,
            "rsi_1h": rsi_1h,
            "confidence": decision.confidence,
            "regime": packet.market_regime,
        })

    return LlmReplayResult(symbol, decisions, no_trades, errors, gated, trades)


async def run_llm_replay(
    data_dir: Path,
    symbols: list[str],
    samples_per_symbol: int,
    settings: Settings | None = None,
    apply_gates: bool = True,
) -> list[dict]:
    """Replay the model across symbols; return the pooled trade list."""
    settings = settings or get_settings()
    cfg = BacktestConfig.from_settings(settings)
    ai = AiClient(settings)
    pooled: list[dict] = []
    try:
        for symbol in symbols:
            try:
                res = await replay_symbol(
                    symbol, data_dir, ai, cfg, samples_per_symbol, settings, apply_gates,
                )
            except Exception as exc:  # noqa: BLE001 — skip a symbol, keep going
                logger.warning("replay failed for %s: %s", symbol, exc)
                continue
            traded = len(res.trades)
            print(
                f"  {symbol:<14} decisions={res.decisions:<4} "
                f"traded={traded:<4} no_trade={res.no_trades:<4} "
                f"gated={res.gated:<4} err={res.errors}"
            )
            pooled.extend(res.trades)
    finally:
        await ai.close()
    return pooled


def _report(pooled: list[dict], cfg: BacktestConfig) -> str:
    if not pooled:
        return "No trades — the model declined every sampled decision point."
    rs = [t["r_multiple"] for t in pooled]
    n = len(rs)
    wins = sum(1 for r in rs if r > 0)
    exp = sum(rs) / n
    std = (sum((r - exp) ** 2 for r in rs) / n) ** 0.5 if n > 1 else 0.0
    gross_win = sum(r for r in rs if r > 0)
    gross_loss = abs(sum(r for r in rs if r < 0))
    avg_rr = sum(t["rr"] for t in pooled) / n
    avg_conf = sum(t["confidence"] for t in pooled) / n
    breakeven_wr = 1.0 / (1.0 + avg_rr) if avg_rr > 0 else 0.0

    verdict = (
        "POSITIVE — the model's calls beat break-even on this sample."
        if exp > 0 else
        "NOT POSITIVE — the model's calls do not beat break-even on this sample."
    )
    lines = [
        "=== LLM REPLAY (live model over history) ===",
        f"Trades taken:    {n}  ({wins}W / {n - wins}L)",
        f"Win rate:        {wins / n * 100:.1f}%   (break-even at avg RR = {breakeven_wr * 100:.1f}%)",
        f"Avg RR chosen:   {avg_rr:.2f}",
        f"Avg confidence:  {avg_conf:.2f}",
        f"Expectancy:      {exp:+.3f} R/trade   (random mechanical baseline ≈ -0.43 R)",
        f"Std dev:         {std:.3f} R",
        f"Profit factor:   {(gross_win / gross_loss) if gross_loss > 0 else float('inf'):.2f}",
        "",
        f"VERDICT: {verdict}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the live LLM over historical decision points")
    parser.add_argument("--data-dir", default="data/historical")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--samples", type=int, default=20, help="Decision points per symbol")
    parser.add_argument("--no-gates", action="store_true",
                        help="Do NOT apply the live risk-engine geometry gates (raw model tape)")
    parser.add_argument("--dump", default="",
                        help="Write the full per-trade tape to this JSON path (for offline threshold sweeps)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s — %(message)s",
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"error: data dir not found: {data_dir}", file=sys.stderr)
        return 2

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else available_symbols(data_dir)
    )
    settings = get_settings()
    cfg = BacktestConfig.from_settings(settings)

    total = args.samples * len(symbols)
    apply_gates = not args.no_gates
    print(f"Replaying {settings.ai_model} over ~{total} decision points "
          f"({args.samples}/symbol × {len(symbols)} symbols)")
    print(f"Provider: {settings.ai_provider}  |  risk-engine gates: "
          f"{'ON' if apply_gates else 'OFF'}\n")

    pooled = asyncio.run(run_llm_replay(data_dir, symbols, args.samples, settings, apply_gates))
    print()
    print(_report(pooled, cfg))

    if args.dump:
        import json
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(pooled, fh)
        print(f"\nTape written: {args.dump} ({len(pooled)} trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
