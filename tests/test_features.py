"""
Sentinel Trader — Feature Engine Tests.

Mathematically verifies the native pandas/numpy indicator implementations
in ``sentinel.data.features`` against:

1. **Closed-form golden values** — flat windows, straight trends, alternating
   series, and gap candles where the correct answer is exact by construction.
2. **Independent reference implementations** — plain-Python recursive loops
   (and the textbook SMA-seeded Wilder recursions for RSI/ATR) evaluated on
   a seeded 300-candle random walk. Convergence of the recursive forms means
   any seeding difference decays below 1e-9 by the last bar.
3. **Depth defenses** — every indicator must raise ``FeatureError`` when fed
   fewer candles than its minimum.

All data is generated from fixed seeds — these tests can never flake.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from sentinel.data.features import (
    MIN_CANDLES,
    REGIME_HIGH_VOL_CHOP,
    REGIME_LOW_VOL_CHOP,
    REGIME_TRENDING_BEAR,
    REGIME_TRENDING_BULL,
    FeatureError,
    _compute_derivatives,
    _volume_zscore,
    adx,
    atr,
    bollinger,
    compute_features,
    detect_market_regime,
    ema,
    macd,
    order_book_imbalance,
    rsi,
)
from sentinel.data.market import (
    TIMEFRAME_MS,
    Candle,
    MarketSnapshot,
    OhlcvSeries,
    OpenInterestSnapshot,
    OrderBookTop,
    TickerSnapshot,
)

# ---------------------------------------------------------------------------
# Deterministic test data
# ---------------------------------------------------------------------------

N_CANDLES = 300
SEED = 42

#: Anchor all synthetic timestamps to a fixed instant (2026-01-01T00:00:00Z).
END_TS_MS = 1_767_225_600_000


def random_walk_closes(n: int = N_CANDLES, seed: int = SEED) -> np.ndarray:
    """Seeded pseudo-random walk around 100 — strictly positive."""
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=n))
    assert closes.min() > 1.0, "walk drifted too low — adjust scale/seed"
    return closes


@pytest.fixture(scope="module")
def walk() -> pd.Series:
    return pd.Series(random_walk_closes())


def make_series(
    closes: np.ndarray,
    timeframe: str = "1h",
    symbol: str = "BTC/USDT",
    spread: float = 0.5,
) -> OhlcvSeries:
    """Build a valid, strictly time-ascending OhlcvSeries from closes."""
    tf_ms = TIMEFRAME_MS[timeframe]
    n = len(closes)
    candles: list[Candle] = []
    for i, close in enumerate(closes):
        ts = END_TS_MS - (n - 1 - i) * tf_ms
        open_ = float(closes[i - 1]) if i > 0 else float(close)
        candles.append(Candle(
            timestamp_ms=ts,
            open=open_,
            high=max(open_, float(close)) + spread,
            low=min(open_, float(close)) - spread,
            close=float(close),
            volume=1000.0 + (i % 7) * 50.0,
        ))
    return OhlcvSeries(symbol=symbol, timeframe=timeframe, candles=tuple(candles))


# ---------------------------------------------------------------------------
# Reference implementations (independent plain-Python loops)
# ---------------------------------------------------------------------------


def ewm_reference(values: list[float], alpha: float) -> float:
    """Recursive EWM (adjust=False), seeding at the first non-NaN value."""
    y: float | None = None
    for x in values:
        if math.isnan(x):
            continue
        y = x if y is None else (1.0 - alpha) * y + alpha * x
    assert y is not None
    return y


def wilder_rsi_reference(closes: np.ndarray, n: int = 14) -> float:
    """Textbook Wilder RSI: SMA seed of the first n deltas, then recursion."""
    deltas = np.diff(closes)
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)
    avg_gain = float(gains[:n].mean())
    avg_loss = float(losses[:n].mean())
    for i in range(n, len(deltas)):
        avg_gain = (avg_gain * (n - 1) + float(gains[i])) / n
        avg_loss = (avg_loss * (n - 1) + float(losses[i])) / n
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def wilder_atr_reference(series: OhlcvSeries, n: int = 14) -> float:
    """Textbook Wilder ATR: SMA seed of the first n TRs, then recursion."""
    candles = series.candles
    trs: list[float] = [candles[0].high - candles[0].low]
    for prev, cur in zip(candles, candles[1:]):
        trs.append(max(
            cur.high - cur.low,
            abs(cur.high - prev.close),
            abs(cur.low - prev.close),
        ))
    atr_val = float(np.mean(trs[:n]))
    for tr in trs[n:]:
        atr_val = (atr_val * (n - 1) + tr) / n
    return atr_val


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEma:
    def test_flat_series_equals_price(self) -> None:
        flat = pd.Series([123.45] * 250)
        for length in (20, 50, 200):
            assert ema(flat, length).iloc[-1] == pytest.approx(123.45, abs=1e-12)

    def test_linear_ramp_converges_to_known_lag(self) -> None:
        # EMA of a linear ramp lags by exactly (span-1)/2 steps once converged.
        step = 2.0
        ramp = pd.Series(100.0 + step * np.arange(300))
        expected = ramp.iloc[-1] - step * (20 - 1) / 2
        assert ema(ramp, 20).iloc[-1] == pytest.approx(expected, rel=1e-9)

    def test_matches_recursive_reference_on_random_walk(self, walk: pd.Series) -> None:
        for length in (20, 50, 200):
            alpha = 2.0 / (length + 1)
            expected = ewm_reference(list(walk), alpha)
            assert ema(walk, length).iloc[-1] == pytest.approx(expected, rel=1e-12)

    def test_insufficient_depth_raises(self) -> None:
        with pytest.raises(FeatureError):
            ema(pd.Series(np.arange(10, dtype=float)), 20)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRsi:
    def test_flat_window_returns_exactly_50(self) -> None:
        flat = pd.Series([250.0] * 100)
        assert rsi(flat, 14).iloc[-1] == 50.0  # exact, by definition

    def test_pure_uptrend_returns_exactly_100(self) -> None:
        up = pd.Series(100.0 + np.arange(100, dtype=float))
        assert rsi(up, 14).iloc[-1] == 100.0

    def test_pure_downtrend_returns_exactly_0(self) -> None:
        down = pd.Series(500.0 - np.arange(100, dtype=float))
        assert rsi(down, 14).iloc[-1] == 0.0

    def test_bounded_between_0_and_100(self, walk: pd.Series) -> None:
        values = rsi(walk, 14).dropna()
        assert ((values >= 0.0) & (values <= 100.0)).all()

    def test_matches_textbook_wilder_reference(self, walk: pd.Series) -> None:
        # Different seeding (EWM vs SMA) decays by (13/14)^~285 ≈ 1e-9.
        expected = wilder_rsi_reference(walk.to_numpy(), 14)
        assert rsi(walk, 14).iloc[-1] == pytest.approx(expected, abs=1e-6)

    def test_insufficient_depth_raises(self) -> None:
        with pytest.raises(FeatureError):
            rsi(pd.Series(np.arange(14, dtype=float)), 14)  # needs 15


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMacd:
    def test_flat_series_is_all_zero(self) -> None:
        flat = pd.Series([777.0] * 100)
        line, signal, hist = macd(flat)
        assert line.iloc[-1] == pytest.approx(0.0, abs=1e-12)
        assert signal.iloc[-1] == pytest.approx(0.0, abs=1e-12)
        assert hist.iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_histogram_is_line_minus_signal(self, walk: pd.Series) -> None:
        line, signal, hist = macd(walk)
        assert hist.iloc[-1] == pytest.approx(line.iloc[-1] - signal.iloc[-1], abs=1e-12)

    def test_uptrend_macd_is_positive(self) -> None:
        up = pd.Series(100.0 + 2.0 * np.arange(300, dtype=float))
        line, _signal, _hist = macd(up)
        # Converged MACD of a ramp with step s: s × ((slow−fast)/2) = 2 × 7 = 14.
        assert line.iloc[-1] == pytest.approx(14.0, rel=1e-9)

    def test_matches_recursive_reference(self, walk: pd.Series) -> None:
        fast_ref = ewm_reference(list(walk), 2.0 / 13)
        slow_ref = ewm_reference(list(walk), 2.0 / 27)
        line, _signal, _hist = macd(walk)
        assert line.iloc[-1] == pytest.approx(fast_ref - slow_ref, rel=1e-12)

    def test_insufficient_depth_raises(self) -> None:
        with pytest.raises(FeatureError):
            macd(pd.Series(np.arange(30, dtype=float)))  # needs 26 + 9 = 35


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestAtr:
    @staticmethod
    def _hlc(closes: np.ndarray, spread: float) -> tuple[pd.Series, pd.Series, pd.Series]:
        close = pd.Series(closes)
        return close + spread, close - spread, close

    def test_constant_range_equals_exactly_that_range(self) -> None:
        # Flat closes with high = close+1, low = close−1 → TR = 2 every bar.
        high, low, close = self._hlc(np.full(100, 100.0), 1.0)
        assert atr(high, low, close, 14).iloc[-1] == pytest.approx(2.0, abs=1e-12)

    def test_zero_range_yields_zero(self) -> None:
        high, low, close = self._hlc(np.full(50, 100.0), 0.0)
        assert atr(high, low, close, 14).iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_gap_dominates_true_range(self) -> None:
        # Bar 0 closes at 100; bar 1 gaps to low=110/high=112.
        # TR_1 = max(112−110, |112−100|, |110−100|) = 12. With length=1
        # (alpha=1) the ATR equals the latest TR exactly.
        high = pd.Series([100.0, 112.0])
        low = pd.Series([100.0, 110.0])
        close = pd.Series([100.0, 111.0])
        assert atr(high, low, close, 1).iloc[-1] == pytest.approx(12.0, abs=1e-12)

    def test_matches_textbook_wilder_reference(self) -> None:
        series = make_series(random_walk_closes(), timeframe="1h")
        df_high = pd.Series([c.high for c in series.candles])
        df_low = pd.Series([c.low for c in series.candles])
        df_close = pd.Series([c.close for c in series.candles])
        expected = wilder_atr_reference(series, 14)
        assert atr(df_high, df_low, df_close, 14).iloc[-1] == pytest.approx(
            expected, rel=1e-6,
        )

    def test_insufficient_depth_raises(self) -> None:
        high, low, close = self._hlc(np.full(14, 100.0), 1.0)
        with pytest.raises(FeatureError):
            atr(high, low, close, 14)  # needs 15


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollinger:
    def test_flat_series_collapses_to_price(self) -> None:
        flat = pd.Series([100.0] * 50)
        upper, middle, lower = bollinger(flat, 20, 2.0)
        assert upper.iloc[-1] == pytest.approx(100.0, abs=1e-12)
        assert middle.iloc[-1] == pytest.approx(100.0, abs=1e-12)
        assert lower.iloc[-1] == pytest.approx(100.0, abs=1e-12)

    def test_alternating_series_exact_golden_values(self) -> None:
        # Window of ten 101s and ten 99s: mean = 100, population std = 1
        # → upper = 102, lower = 98, all exact in binary floating point.
        closes = pd.Series([101.0 if i % 2 == 0 else 99.0 for i in range(40)])
        upper, middle, lower = bollinger(closes, 20, 2.0)
        assert middle.iloc[-1] == pytest.approx(100.0, abs=1e-12)
        assert upper.iloc[-1] == pytest.approx(102.0, abs=1e-12)
        assert lower.iloc[-1] == pytest.approx(98.0, abs=1e-12)

    def test_matches_numpy_reference(self, walk: pd.Series) -> None:
        tail = walk.to_numpy()[-20:]
        mean, std = float(np.mean(tail)), float(np.std(tail, ddof=0))
        upper, middle, lower = bollinger(walk, 20, 2.0)
        assert middle.iloc[-1] == pytest.approx(mean, rel=1e-12)
        assert upper.iloc[-1] == pytest.approx(mean + 2 * std, rel=1e-12)
        assert lower.iloc[-1] == pytest.approx(mean - 2 * std, rel=1e-12)

    def test_band_ordering_holds_everywhere(self, walk: pd.Series) -> None:
        upper, middle, lower = bollinger(walk, 20, 2.0)
        valid = ~middle.isna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_insufficient_depth_raises(self) -> None:
        with pytest.raises(FeatureError):
            bollinger(pd.Series(np.arange(19, dtype=float)), 20)


# ---------------------------------------------------------------------------
# Volume z-score division-by-zero defense
# ---------------------------------------------------------------------------


class TestVolumeZscore:
    def test_flat_volume_returns_zero(self) -> None:
        assert _volume_zscore(pd.Series([500.0] * 25)) == 0.0

    def test_spike_is_positive(self) -> None:
        volumes = pd.Series([100.0] * 24 + [1000.0])
        assert _volume_zscore(volumes) > 3.0


# ---------------------------------------------------------------------------
# Full-packet integration (deterministic end-to-end)
# ---------------------------------------------------------------------------


def _make_book(last_close: float) -> OrderBookTop:
    """Synthetic 10-level book: bids slightly heavier than asks."""
    best_bid = last_close - 0.05
    best_ask = last_close + 0.05
    bids = tuple(
        (best_bid * (1.0 - 0.0005 * i), 3.0) for i in range(10)
    )
    asks = tuple(
        (best_ask * (1.0 + 0.0005 * i), 2.0) for i in range(10)
    )
    return OrderBookTop(
        best_bid=best_bid, best_ask=best_ask,
        spread_pct=0.01, bid_depth_usdt=500_000.0, ask_depth_usdt=450_000.0,
        bids=bids, asks=asks,
    )


def _make_snapshot(n_candles: int = N_CANDLES) -> MarketSnapshot:
    closes = random_walk_closes(n_candles)
    last_close = float(closes[-1])
    series = {
        tf: make_series(closes, timeframe=tf) for tf in ("15m", "1h", "4h")
    }
    return MarketSnapshot(
        symbol="BTC/USDT",
        swap_symbol="BTC/USDT:USDT",
        fetched_at_ms=END_TS_MS + 5_000,
        series=series,
        ticker=TickerSnapshot(
            symbol="BTC/USDT", last=last_close,
            bid=last_close - 0.05, ask=last_close + 0.05,
            timestamp_ms=END_TS_MS + 5_000,
        ),
        funding_rate_8h=0.0001,
        order_book=_make_book(last_close),
    )


class TestComputeFeatures:
    def test_packet_is_complete_and_schema_shaped(self) -> None:
        packet = compute_features(_make_snapshot())
        data = packet.to_dict()
        assert set(data["timeframes"].keys()) == {"15m", "1h", "4h"}
        for tf_data in data["timeframes"].values():
            assert len(tf_data["ohlcv_last_20"]) == 20
            for key in ("ema_20", "ema_50", "ema_200", "rsi_14", "atr_14",
                        "volume_zscore", "return_24h_pct", "return_7d_pct"):
                assert math.isfinite(tf_data[key]), f"{key} not finite"
        assert packet.atr_14_1h == packet.timeframes["1h"].atr_14

        # institutional features must be present in the AI payload
        assert data["market_regime"] in {
            REGIME_TRENDING_BULL, REGIME_TRENDING_BEAR,
            REGIME_HIGH_VOL_CHOP, REGIME_LOW_VOL_CHOP,
        }
        for key in ("obi_0_1pct", "obi_0_5pct", "obi_1_0pct"):
            assert -1.0 <= data["order_book_top"][key] <= 1.0
        assert set(data["derivatives"]) == {
            "open_interest", "oi_change_24h_pct",
            "volume_to_oi_ratio", "is_liquidation_cascade_likely",
        }

    def test_packet_json_is_byte_deterministic(self) -> None:
        a = compute_features(_make_snapshot()).to_json()
        b = compute_features(_make_snapshot()).to_json()
        assert a == b

    def test_shallow_series_raises_feature_error(self) -> None:
        with pytest.raises(FeatureError):
            compute_features(_make_snapshot(n_candles=MIN_CANDLES - 1))


# ---------------------------------------------------------------------------
# Market regime classifier
# ---------------------------------------------------------------------------


def _regime_frame(close: np.ndarray, spread: np.ndarray | float = 0.5) -> pd.DataFrame:
    spread_arr = np.broadcast_to(np.asarray(spread, dtype=float), close.shape)
    return pd.DataFrame({
        "high": close + spread_arr,
        "low": close - spread_arr,
        "close": close,
    })


class TestMarketRegime:
    def test_steady_ramp_up_is_trending_bull(self) -> None:
        close = 100.0 + np.arange(150, dtype=float)
        assert detect_market_regime(_regime_frame(close)) == REGIME_TRENDING_BULL

    def test_steady_ramp_down_is_trending_bear(self) -> None:
        close = 400.0 - np.arange(150, dtype=float)
        assert detect_market_regime(_regime_frame(close)) == REGIME_TRENDING_BEAR

    def test_quiet_flat_market_is_low_vol_chop(self) -> None:
        # Tiny alternating wiggle: no direction, no volatility expansion.
        close = 100.0 + 0.1 * np.where(np.arange(150) % 2 == 0, 1.0, -1.0)
        assert detect_market_regime(_regime_frame(close, 0.2)) == REGIME_LOW_VOL_CHOP

    def test_volatility_expansion_without_trend_is_high_vol_chop(self) -> None:
        # Alternating series (zero net direction) whose amplitude explodes
        # over the last 15 candles: ADX stays low, ATR% spikes vs its median.
        n = 150
        amp = np.where(np.arange(n) < n - 15, 0.5, 6.0)
        sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        close = 100.0 + amp * sign
        frame = _regime_frame(close, amp)
        assert detect_market_regime(frame) == REGIME_HIGH_VOL_CHOP

    def test_insufficient_depth_raises(self) -> None:
        close = 100.0 + np.arange(50, dtype=float)
        with pytest.raises(FeatureError):
            detect_market_regime(_regime_frame(close))

    def test_adx_extremes(self) -> None:
        # Perfect one-way ramp: DI- == 0, so DX == 100 on every bar → ADX ≈ 100.
        close = pd.Series(100.0 + np.arange(120, dtype=float))
        high, low = close + 0.5, close - 0.5
        adx_series, di_plus, di_minus = adx(high, low, close, 14)
        assert adx_series.iloc[-1] > 90.0
        assert di_plus.iloc[-1] > di_minus.iloc[-1]

        # Perfectly flat tape: zero directional movement → ADX == 0, no NaN/inf.
        flat = pd.Series(np.full(120, 100.0))
        adx_flat, dip, dim = adx(flat + 0.5, flat - 0.5, flat, 14)
        assert adx_flat.iloc[-1] == 0.0
        assert dip.iloc[-1] == 0.0 and dim.iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# Order book imbalance
# ---------------------------------------------------------------------------


class TestOrderBookImbalance:
    def test_golden_value(self) -> None:
        bids = ((99.9, 3.0),)
        asks = ((100.1, 1.0),)
        assert order_book_imbalance(bids, asks, 100.0, 0.01) == pytest.approx(0.5)

    def test_all_bids_is_plus_one(self) -> None:
        assert order_book_imbalance(((99.9, 5.0),), (), 100.0, 0.01) == 1.0

    def test_all_asks_is_minus_one(self) -> None:
        assert order_book_imbalance((), ((100.1, 5.0),), 100.0, 0.01) == -1.0

    def test_empty_band_is_neutral_not_division_by_zero(self) -> None:
        # Levels exist but all sit outside the ±0.05% band.
        bids = ((99.0, 5.0),)
        asks = ((101.0, 5.0),)
        assert order_book_imbalance(bids, asks, 100.0, 0.0005) == 0.0

    def test_band_widening_includes_more_levels(self) -> None:
        bids = ((99.95, 1.0), (99.50, 10.0))   # second level only in the 1% band
        asks = ((100.05, 1.0),)
        narrow = order_book_imbalance(bids, asks, 100.0, 0.001)
        wide = order_book_imbalance(bids, asks, 100.0, 0.01)
        assert narrow == pytest.approx(0.0)            # 1 vs 1
        assert wide == pytest.approx((11.0 - 1.0) / 12.0)


# ---------------------------------------------------------------------------
# Derivatives / liquidation-fuel features
# ---------------------------------------------------------------------------


class TestDerivativesFeatures:
    def test_no_oi_data_degrades_to_none(self) -> None:
        deriv = _compute_derivatives(_make_snapshot())
        assert deriv.open_interest is None
        assert deriv.oi_change_24h_pct is None
        assert deriv.volume_to_oi_ratio is None
        assert deriv.is_liquidation_cascade_likely is False

    def test_cascade_flag_fires_on_oi_flush_with_extreme_churn(self) -> None:
        snap = _make_snapshot()
        vol_24h = sum(c.volume for c in snap.series["1h"].candles[-24:])
        oi_now = vol_24h / 2.0                  # ratio = 2.0 ≥ 1.5
        oi_past = oi_now / 0.90                 # change ≈ −10% ≤ −5%
        snap = dataclasses.replace(
            snap,
            open_interest=OpenInterestSnapshot(
                symbol="BTC/USDT", amount=oi_now,
                value_usdt=None, timestamp_ms=END_TS_MS,
            ),
            open_interest_24h_ago=oi_past,
        )
        deriv = _compute_derivatives(snap)
        assert deriv.oi_change_24h_pct == pytest.approx(-10.0, abs=0.01)
        assert deriv.volume_to_oi_ratio == pytest.approx(2.0, abs=0.001)
        assert deriv.is_liquidation_cascade_likely is True

    def test_growing_oi_never_flags_cascade(self) -> None:
        snap = _make_snapshot()
        vol_24h = sum(c.volume for c in snap.series["1h"].candles[-24:])
        oi_now = vol_24h / 2.0
        snap = dataclasses.replace(
            snap,
            open_interest=OpenInterestSnapshot(
                symbol="BTC/USDT", amount=oi_now,
                value_usdt=None, timestamp_ms=END_TS_MS,
            ),
            open_interest_24h_ago=oi_now * 0.80,    # OI grew 25%
        )
        deriv = _compute_derivatives(snap)
        assert deriv.oi_change_24h_pct is not None and deriv.oi_change_24h_pct > 0
        assert deriv.is_liquidation_cascade_likely is False
