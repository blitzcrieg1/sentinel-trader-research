"""
Sentinel Trader — Offline Fine-Tuning Dataset Harvester.

Turns the bot's own audit trail into a supervised fine-tuning dataset for
a custom Gemini model:

    feature_snapshots (SQLite)  →  "what did the market do 2h later?"
                                →  label: long / short / no_trade
                                →  training_data.jsonl (Gemini format)

Each output line pairs the *exact* features JSON the live model saw with
the decision a perfect oracle would have made, formatted as::

    {"messages": [
        {"role": "user",  "content": "<raw features_json>"},
        {"role": "model", "content": "<schema-valid AiDecision JSON>"}
    ]}

Design rules:
- **Offline & read-only.** The database is opened in read-only mode; this
  script can never corrupt live state. Run it while the bot is running or
  stopped — it doesn't matter.
- **One paginated history fetch per symbol**, not one request per snapshot.
  Snapshots cluster heavily per symbol, so we pull each symbol's full 1h
  candle range once and label from an in-memory lookup. CCXT's built-in
  throttler (``enableRateLimit``) plus bounded retries handle rate limits.
- **Labels are contract-valid.** Every generated completion is validated
  through the live ``AiDecision`` pydantic model before it is written.
  A fine-tune must never teach the model to violate its own schema.

Run with::

    python -m sentinel.ai.trainer
    python -m sentinel.ai.trainer --db data/sentinel.sqlite --output training_data.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import aiosqlite
import ccxt.async_support as ccxt
from tqdm import tqdm

from sentinel.ai.contract import AiDecision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: How far into the future we look to judge the setup.
DEFAULT_HORIZON_HOURS: Final[int] = 2

#: Move threshold (in %) that separates a directional label from no_trade.
DEFAULT_THRESHOLD_PCT: Final[float] = 1.0

#: 1h candles per pagination page. Conservative — well under MEXC's cap.
CANDLES_PER_PAGE: Final[int] = 500

#: Bounded retries for a single candle page on transient network errors.
MAX_FETCH_RETRIES: Final[int] = 4

#: Stop-loss / take-profit distances for the synthetic labels, in ATR(14, 1h)
#: multiples. These mirror sane live behaviour: SL outside the noise band,
#: TP at 2 ATR for a ~1.33 reward:risk.
SL_ATR_MULTIPLE: Final[float] = 1.5
TP_ATR_MULTIPLE: Final[float] = 2.0

#: Fallback SL/TP distance (% of price) when ATR is missing from a snapshot.
FALLBACK_DISTANCE_PCT: Final[float] = 1.0

_MS_PER_HOUR: Final[int] = 60 * 60 * 1000


# ---------------------------------------------------------------------------
# Row containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SnapshotRow:
    """One feature_snapshots row, with its timestamp parsed to epoch-ms."""

    symbol: str
    features_json: str
    created_at_ms: int


@dataclass(slots=True)
class LabeledExample:
    """A finished training example, ready for JSONL serialization."""

    user_content: str     # the raw features_json string, untouched
    model_content: str    # compact, schema-valid AiDecision JSON


# ---------------------------------------------------------------------------
# Step 1 — load snapshots from SQLite (read-only)
# ---------------------------------------------------------------------------


def _parse_utc_timestamp(raw: str) -> int | None:
    """Parse the store's ISO timestamp ('%Y-%m-%dT%H:%M:%SZ') to epoch-ms.

    Returns None on garbage — a snapshot we cannot place in time cannot
    be labeled and is skipped, never guessed.
    """
    try:
        dt = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        try:
            # Defensive fallback for any other ISO-8601 variant.
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    return int(dt.timestamp() * 1000)


async def load_snapshots(db_path: Path) -> list[SnapshotRow]:
    """Fetch every feature_snapshots row. Opens the DB strictly read-only."""
    # mode=ro guarantees this offline script can never write to live state.
    uri = f"file:{db_path.as_posix()}?mode=ro"
    rows: list[SnapshotRow] = []
    dropped = 0

    async with aiosqlite.connect(uri, uri=True) as db:
        cursor = await db.execute(
            "SELECT symbol, features_json, created_at "
            "FROM feature_snapshots ORDER BY symbol, created_at"
        )
        async for symbol, features_json, created_at in cursor:
            ts_ms = _parse_utc_timestamp(created_at)
            if ts_ms is None or not features_json:
                dropped += 1
                continue
            rows.append(SnapshotRow(
                symbol=str(symbol),
                features_json=str(features_json),
                created_at_ms=ts_ms,
            ))

    if dropped:
        logger.warning("dropped %d snapshots with unparseable timestamp/empty JSON", dropped)
    logger.info("loaded %d feature snapshots from %s", len(rows), db_path)
    return rows


# ---------------------------------------------------------------------------
# Step 2 — fetch historical 1h candles (one paginated sweep per symbol)
# ---------------------------------------------------------------------------


class CandleHistory:
    """Paginated historical 1h candle fetcher with an open-time lookup map.

    Rate-limit strategy:
    - ``enableRateLimit=True`` makes CCXT's internal throttler space out
      requests to the exchange's documented limit automatically.
    - On transient network errors / explicit rate-limit responses we back
      off exponentially, bounded by ``MAX_FETCH_RETRIES``.
    """

    def __init__(self) -> None:
        self._exchange = ccxt.mexc({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        # symbol -> {candle_open_ts_ms: (open, close)}
        self._candles: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)

    async def close(self) -> None:
        await self._exchange.close()

    @staticmethod
    def _to_swap(symbol: str) -> str:
        return f"{symbol.replace(':USDT', '').strip()}:USDT"

    def ingest_1h_candles(self, symbol: str, candles: list[Candle]) -> None:
        """Seed the oracle from pre-fetched 1h candles (avoids a second MEXC sweep)."""
        for candle in candles:
            if candle.open > 0 and candle.close > 0:
                self._candles[symbol][candle.timestamp_ms] = (candle.open, candle.close)

    async def _fetch_page(self, swap: str, since_ms: int) -> list[list[Any]]:
        """One candle page with bounded exponential-backoff retries."""
        last_error: Exception | None = None
        for attempt in range(1, MAX_FETCH_RETRIES + 1):
            try:
                return await self._exchange.fetch_ohlcv(
                    swap, "1h", since=since_ms, limit=CANDLES_PER_PAGE,
                )
            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as exc:
                # Explicit rate-limit pushback: wait noticeably longer.
                last_error = exc
                await asyncio.sleep(5.0 * attempt)
            except (ccxt.NetworkError, asyncio.TimeoutError) as exc:
                last_error = exc
                await asyncio.sleep(2.0 ** attempt)
        raise RuntimeError(
            f"candle page fetch failed after {MAX_FETCH_RETRIES} attempts: {last_error}"
        )

    async def load_range(self, symbol: str, start_ms: int, end_ms: int) -> None:
        """Pull all 1h candles covering [start, end] into the lookup map."""
        swap = self._to_swap(symbol)
        cursor = start_ms

        while cursor <= end_ms:
            page = await self._fetch_page(swap, cursor)
            if not page:
                break  # exchange has nothing beyond this point

            for row in page:
                # Defensive parse: a single dirty row must not kill the run.
                try:
                    ts, open_, close = int(row[0]), float(row[1]), float(row[4])
                except (TypeError, ValueError, IndexError):
                    continue
                if open_ > 0 and close > 0:
                    self._candles[symbol][ts] = (open_, close)

            newest = max(int(r[0]) for r in page)
            if newest <= cursor:
                break  # no forward progress — avoid an infinite loop
            cursor = newest + _MS_PER_HOUR

    def price_at(self, symbol: str, ts_ms: int) -> float | None:
        """Price at an hour boundary = the open of the candle starting there."""
        bucket = ts_ms - (ts_ms % _MS_PER_HOUR)
        candle = self._candles[symbol].get(bucket)
        return candle[0] if candle is not None else None


# ---------------------------------------------------------------------------
# Step 3 — label one snapshot
# ---------------------------------------------------------------------------


def _label_from_move(change_pct: float, threshold_pct: float) -> str:
    """The core labeling rule: ±threshold% over the horizon."""
    if change_pct > threshold_pct:
        return "long"
    if change_pct < -threshold_pct:
        return "short"
    return "no_trade"


def _extract_price_and_atr(features: dict[str, Any]) -> tuple[float | None, float | None]:
    """Pull the baseline price and ATR(14, 1h) out of a FeaturePacket dict."""
    price = features.get("current_price")
    atr = (features.get("timeframes") or {}).get("1h", {}).get("atr_14")
    price_f = float(price) if isinstance(price, (int, float)) and price > 0 else None
    atr_f = float(atr) if isinstance(atr, (int, float)) and atr > 0 else None
    return price_f, atr_f


def build_model_content(label: str, symbol: str, price: float, atr: float | None) -> str | None:
    """Build the oracle completion as a compact, schema-valid JSON string.

    Returns None if the constructed decision fails the live ``AiDecision``
    contract — we refuse to put schema-invalid lines in the dataset.
    """
    # SL/TP distances: ATR-based when available, % fallback otherwise.
    distance = atr if atr is not None else price * (FALLBACK_DISTANCE_PCT / 100.0)

    if label == "long":
        sl = round(price - SL_ATR_MULTIPLE * distance, 8)
        tp = round(price + TP_ATR_MULTIPLE * distance, 8)
    elif label == "short":
        sl = round(price + SL_ATR_MULTIPLE * distance, 8)
        tp = round(price - TP_ATR_MULTIPLE * distance, 8)
    else:
        # no_trade: SL/TP are never acted upon; use the same placeholder
        # convention as contract.no_trade_decision().
        sl, tp = 1.0, 1.0

    decision = {
        "schema_version": "1.0",
        "symbol": symbol,
        "decision": label,
        "confidence": 0.9,
        "timeframe_alignment": "aligned",
        "entry": {"type": "market", "limit_price": None},
        "stop_loss_price": sl,
        "take_profit_prices": [tp],
        "invalidation": "Generated by offline trainer.",
        "rationale": "Based on statistical backtesting of these exact features.",
        "risk_flags": [],
    }

    try:
        AiDecision.model_validate(decision)
    except Exception as exc:  # noqa: BLE001 — any contract violation = skip the row
        logger.warning("generated label failed contract validation (%s): %s", symbol, exc)
        return None

    return json.dumps(decision, separators=(",", ":"))


def label_snapshot(
    snap: SnapshotRow,
    history: CandleHistory,
    horizon_hours: int,
    threshold_pct: float,
) -> tuple[LabeledExample | None, str]:
    """Label one snapshot. Returns (example | None, outcome-tag for stats)."""
    # Parse the stored features JSON. It is the literal payload the live
    # model saw, so it is also the literal "user" side of the example.
    try:
        features = json.loads(snap.features_json)
    except json.JSONDecodeError:
        return None, "skipped:bad_json"

    base_price, atr = _extract_price_and_atr(features)
    if base_price is None:
        # No usable baseline inside the snapshot — try the candle itself.
        base_price = history.price_at(snap.symbol, snap.created_at_ms)
        if base_price is None:
            return None, "skipped:no_base_price"

    # Snapshots fire seconds after a candle close, so flooring the snapshot
    # time to its hour bucket and stepping forward gives exact boundaries.
    snapshot_bucket = snap.created_at_ms - (snap.created_at_ms % _MS_PER_HOUR)
    future_price = history.price_at(snap.symbol, snapshot_bucket + horizon_hours * _MS_PER_HOUR)
    if future_price is None:
        # Snapshot too recent (future not written yet) or a candle gap.
        return None, "skipped:no_future_candle"

    change_pct = (future_price - base_price) / base_price * 100.0
    label = _label_from_move(change_pct, threshold_pct)

    model_content = build_model_content(label, snap.symbol, base_price, atr)
    if model_content is None:
        return None, "skipped:contract_invalid"

    return LabeledExample(snap.features_json, model_content), f"labeled:{label}"


# ---------------------------------------------------------------------------
# Step 4 — orchestrate and write JSONL
# ---------------------------------------------------------------------------


async def harvest(
    db_path: Path,
    output_path: Path,
    horizon_hours: int,
    threshold_pct: float,
) -> Counter:
    """Full pipeline: load → fetch history → label → write JSONL."""
    snapshots = await load_snapshots(db_path)
    if not snapshots:
        logger.warning("no feature snapshots found — nothing to harvest")
        return Counter()

    # Per-symbol time ranges so each symbol needs exactly one history sweep.
    ranges: dict[str, tuple[int, int]] = {}
    for snap in snapshots:
        lo, hi = ranges.get(snap.symbol, (snap.created_at_ms, snap.created_at_ms))
        ranges[snap.symbol] = (min(lo, snap.created_at_ms), max(hi, snap.created_at_ms))

    history = CandleHistory()
    stats: Counter = Counter()
    try:
        # ---- fetch candle history, one symbol at a time (rate-limit safe) --
        for symbol, (lo, hi) in tqdm(
            sorted(ranges.items()), desc="Fetching 1h history", unit="symbol",
        ):
            # Pad the range: 1h before the first snapshot, horizon + 1h after
            # the last, so every lookup this run needs is guaranteed covered.
            await history.load_range(
                symbol,
                start_ms=lo - _MS_PER_HOUR,
                end_ms=hi + (horizon_hours + 1) * _MS_PER_HOUR,
            )

        # ---- label every snapshot and stream lines straight to disk --------
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as out:
            for snap in tqdm(snapshots, desc="Labeling snapshots", unit="row"):
                example, outcome = label_snapshot(snap, history, horizon_hours, threshold_pct)
                stats[outcome] += 1
                if example is None:
                    continue
                line = {
                    "messages": [
                        {"role": "user", "content": example.user_content},
                        {"role": "model", "content": example.model_content},
                    ]
                }
                out.write(json.dumps(line, separators=(",", ":")) + "\n")
    finally:
        await history.close()

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _default_db_path() -> Path:
    """Resolve the live DB path from settings; fall back to the spec default."""
    try:
        from sentinel.config import get_settings
        return Path(get_settings().db_path)
    except Exception:  # noqa: BLE001 — offline tool must run without a full .env
        return Path("data") / "sentinel.sqlite"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sentinel.ai.trainer",
        description="Harvest feature snapshots into a Gemini fine-tuning JSONL dataset.",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="Path to sentinel.sqlite (default: from settings, else data/sentinel.sqlite)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("training_data.jsonl"),
        help="Output JSONL path (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS,
        help=f"Look-ahead horizon in hours (default: {DEFAULT_HORIZON_HOURS})",
    )
    parser.add_argument(
        "--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT,
        help=f"Move %% separating long/short from no_trade (default: {DEFAULT_THRESHOLD_PCT})",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)

    db_path: Path = args.db if args.db is not None else _default_db_path()
    if not db_path.exists():
        logger.error("database not found: %s", db_path)
        return 1

    stats = await harvest(
        db_path=db_path,
        output_path=args.output,
        horizon_hours=args.horizon_hours,
        threshold_pct=args.threshold_pct,
    )

    # ---- final report -----------------------------------------------------
    written = sum(v for k, v in stats.items() if k.startswith("labeled:"))
    print("\n=== Harvest complete ===")
    for key in sorted(stats):
        print(f"  {key:<28} {stats[key]}")
    print(f"  {'total written':<28} {written}")
    print(f"  output: {args.output.resolve()}")

    if written and stats.get("labeled:no_trade", 0) == written:
        # A dataset that is 100% no_trade teaches the model to do nothing.
        print("\nWARNING: every example is no_trade — consider a lower --threshold-pct.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
