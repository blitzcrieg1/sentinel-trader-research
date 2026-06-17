"""
Sentinel Trader — Time Machine Backfill Generator.

Generates a supervised fine-tuning dataset by replaying the bot's feature
pipeline over historical OHLCV.  Uses ``data/historical/`` cache when present,
otherwise downloads from MEXC via the hardened historical fetcher.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import logging
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sentinel.config import get_settings
from sentinel.data.features import compute_features
from sentinel.data.historical import (
    DEFAULT_TIMEFRAMES,
    MexcHistoricalFetcher,
    load_cached_candles,
)
from sentinel.data.market import (
    Candle,
    MarketSnapshot,
    OhlcvSeries,
    OrderBookTop,
    TickerSnapshot,
)
from sentinel.ai.trainer import CandleHistory, SnapshotRow, label_snapshot

logger = logging.getLogger(__name__)

_MS_PER_15M = 15 * 60 * 1000
_MS_PER_1H = 60 * 60 * 1000
_MS_PER_4H = 4 * 60 * 60 * 1000


class CandleStore:
    """In-memory OHLCV store backed by local cache and/or live MEXC fetch."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._fetcher: MexcHistoricalFetcher | None = None
        self.candles: dict[str, dict[str, dict[int, Candle]]] = {}
        self.arrays: dict[str, dict[str, tuple[list[int], list[Candle]]]] = {}

    async def close(self) -> None:
        if self._fetcher is not None:
            await self._fetcher.close()
            self._fetcher = None

    async def load_range(
        self,
        symbol: str,
        timeframes: list[str],
        start_ms: int,
        end_ms: int,
    ) -> None:
        if symbol not in self.candles:
            self.candles[symbol] = {tf: {} for tf in timeframes}

        for tf in timeframes:
            candles: list[Candle] | None = None
            if self._cache_dir is not None:
                candles = load_cached_candles(self._cache_dir, symbol, tf)
                if candles:
                    logger.info(
                        "loaded %d %s candles for %s from cache", len(candles), tf, symbol,
                    )

            if candles is None:
                if self._fetcher is None:
                    self._fetcher = MexcHistoricalFetcher()
                    await self._fetcher.connect()
                logger.info("fetching %s %s from MEXC", symbol, tf)
                candles = await self._fetcher.fetch_range(symbol, tf, start_ms, end_ms)

            for candle in candles:
                if candle.timestamp_ms <= end_ms:
                    self.candles[symbol][tf][candle.timestamp_ms] = candle

    def finalize_arrays(self, symbol: str) -> None:
        self.arrays[symbol] = {}
        for tf in DEFAULT_TIMEFRAMES:
            c_list = list(self.candles[symbol][tf].values())
            c_list.sort(key=lambda c: c.timestamp_ms)
            ts_list = [c.timestamp_ms for c in c_list]
            self.arrays[symbol][tf] = (ts_list, c_list)

    def candles_1h_list(self, symbol: str) -> list[Candle]:
        return list(self.candles[symbol]["1h"].values())


def build_mock_snapshot(symbol: str, target_ms: int, store: CandleStore) -> MarketSnapshot | None:
    """Build a MarketSnapshot mimicking what the live bot sees at ``target_ms``."""
    series_dict: dict[str, OhlcvSeries] = {}

    for tf in DEFAULT_TIMEFRAMES:
        ts_array, candle_array = store.arrays[symbol][tf]
        idx = bisect.bisect_right(ts_array, target_ms)
        if idx < 201:
            return None
        start_idx = max(0, idx - 300)
        series_dict[tf] = OhlcvSeries(
            symbol=symbol, timeframe=tf, candles=tuple(candle_array[start_idx:idx]),
        )

    last_15m = series_dict["15m"].candles[-1]
    if last_15m.timestamp_ms != target_ms:
        return None

    close_price = last_15m.close
    mock_book = OrderBookTop(
        best_bid=close_price * 0.9999,
        best_ask=close_price * 1.0001,
        spread_pct=0.02,
        bid_depth_usdt=100_000.0,
        ask_depth_usdt=100_000.0,
        bids=((close_price * 0.9999, 100.0),),
        asks=((close_price * 1.0001, 100.0),),
    )
    mock_ticker = TickerSnapshot(
        symbol=symbol,
        last=close_price,
        bid=mock_book.best_bid,
        ask=mock_book.best_ask,
        timestamp_ms=target_ms,
    )
    return MarketSnapshot(
        symbol=symbol,
        swap_symbol=f"{symbol.replace(':USDT', '')}:USDT",
        fetched_at_ms=target_ms,
        series=series_dict,
        ticker=mock_ticker,
        order_book=mock_book,
        open_interest=None,
        open_interest_24h_ago=None,
        funding_rate_8h=0.0001,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AI training dataset from historical data.")
    parser.add_argument("--days", type=int, default=30, help="Days of data to backfill (default: 30)")
    parser.add_argument("--step-minutes", type=int, default=15, help="Step size in minutes (default: 15)")
    parser.add_argument("--output", type=str, default="backfill_dataset.jsonl", help="Output file")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/historical",
        help="Use exported OHLCV JSON from sentinel.data.historical (default: data/historical)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Always download from MEXC instead of using local cache",
    )
    args = parser.parse_args()

    settings = get_settings()
    symbols = settings.scan_symbols
    cache_dir = None if args.no_cache else Path(args.cache_dir)
    if cache_dir is not None and not cache_dir.exists():
        logger.warning("cache dir %s missing — will fetch from MEXC", cache_dir)
        cache_dir = None

    now = datetime.now(UTC)
    end_dt = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    end_ms = int(end_dt.timestamp() * 1000)

    backfill_start_ms = end_ms - (args.days * 24 * 60 * 60 * 1000)
    warmup_margin_ms = 35 * 24 * 60 * 60 * 1000
    fetch_start_ms = backfill_start_ms - warmup_margin_ms

    store = CandleStore(cache_dir=cache_dir)
    oracle = CandleHistory()
    stats: Counter[str] = Counter()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step_ms = args.step_minutes * 60 * 1000

    try:
        with out_path.open("w", encoding="utf-8") as out_file:
            for idx_sym, symbol in enumerate(symbols, 1):
                print(f"\n--- [{idx_sym}/{len(symbols)}] {symbol} ---")

                await store.load_range(symbol, list(DEFAULT_TIMEFRAMES), fetch_start_ms, end_ms)
                store.finalize_arrays(symbol)

                oracle.ingest_1h_candles(symbol, store.candles_1h_list(symbol))

                cursor_ms = backfill_start_ms - (backfill_start_ms % step_ms)
                total_iters = max(1, (end_ms - backfill_start_ms) // step_ms)
                iters_done = 0
                examples_found = 0
                start_time = time.time()

                while cursor_ms <= end_ms:
                    snapshot = build_mock_snapshot(symbol, cursor_ms, store)
                    if snapshot is not None:
                        try:
                            feature_packet = compute_features(snapshot)
                            snap_row = SnapshotRow(
                                symbol=symbol,
                                features_json=feature_packet.to_json(),
                                created_at_ms=cursor_ms,
                            )
                            example, outcome = label_snapshot(
                                snap=snap_row,
                                history=oracle,
                                horizon_hours=2,
                                threshold_pct=1.0,
                            )
                            stats[outcome] += 1
                            if example is not None:
                                examples_found += 1
                                line = {
                                    "messages": [
                                        {"role": "user", "content": example.user_content},
                                        {"role": "model", "content": example.model_content},
                                    ]
                                }
                                out_file.write(json.dumps(line, separators=(",", ":")) + "\n")
                        except Exception:
                            stats["feature_computation_error"] += 1
                    else:
                        stats["insufficient_candles"] += 1

                    cursor_ms += step_ms
                    iters_done += 1

                    if iters_done % 100 == 0 or iters_done == total_iters:
                        pct = (iters_done / total_iters) * 100
                        elapsed = time.time() - start_time
                        eta = (elapsed / iters_done) * (total_iters - iters_done) if iters_done else 0
                        print(
                            f"\r  {symbol}: {pct:.1f}% ({iters_done}/{total_iters}) "
                            f"examples={examples_found} eta={eta / 60:.1f}m",
                            end="",
                        )
                        sys.stdout.flush()

                print()
    finally:
        await store.close()
        await oracle.close()

    print("\n=== Backfill Generation Complete ===")
    for key, value in sorted(stats.items()):
        print(f"  {key:<28} {value}")
    print(f"\nOutput saved to: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("sentinel.data.features").setLevel(logging.WARNING)
    raise SystemExit(asyncio.run(main()))
