"""
Backtest — performance metrics.

Turns a pooled list of ``BacktestTrade`` (R-multiples) into the standard
edge-evaluation statistics: win rate, expectancy, profit factor, an equity
curve with max drawdown, and an annualised Sharpe estimated from trade
frequency. Also breaks results down per regime, because a strategy that
only works in one regime is a very different thing from one that works
across all of them.

Equity model: trades are applied in entry-time order to a single equity
curve, each risking ``risk_per_trade_pct`` of *current* equity at −1R. This
is a portfolio approximation (overlapping positions are sequenced, not
margined jointly) — honest enough to reveal whether an edge exists, not a
substitute for live position accounting.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from sentinel.backtest.engine import BacktestTrade

_MS_PER_YEAR = 365.25 * 24 * 60 * 60 * 1000


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    expectancy_r: float          # mean R per trade
    std_r: float
    profit_factor: float
    total_return_pct: float      # on the compounded equity curve
    max_drawdown_pct: float
    sharpe_annualised: float
    avg_bars_held: float
    span_days: float
    trades_per_week: float
    by_regime: dict[str, dict[str, float]] = field(default_factory=dict)
    by_outcome: dict[str, int] = field(default_factory=dict)

    def format_text(self) -> str:
        """Human-readable report block."""
        lines = [
            "=== BACKTEST PERFORMANCE ===",
            f"Trades:          {self.n_trades}  ({self.wins}W / {self.losses}L)",
            f"Win rate:        {self.win_rate * 100:.1f}%",
            f"Expectancy:      {self.expectancy_r:+.3f} R/trade",
            f"Std dev:         {self.std_r:.3f} R",
            f"Profit factor:   {self.profit_factor:.2f}",
            f"Total return:    {self.total_return_pct:+.1f}%  (compounded)",
            f"Max drawdown:    {self.max_drawdown_pct:.1f}%",
            f"Sharpe (annual): {self.sharpe_annualised:.2f}",
            f"Avg hold:        {self.avg_bars_held:.0f} bars "
            f"({self.avg_bars_held * 15 / 60:.1f}h)",
            f"Span:            {self.span_days:.0f} days "
            f"({self.trades_per_week:.1f} trades/week)",
            "",
            "By outcome:      "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.by_outcome.items())),
        ]
        if self.by_regime:
            lines.append("")
            lines.append("By regime:")
            for regime, stats in sorted(self.by_regime.items()):
                lines.append(
                    f"  {regime:<16} n={int(stats['n']):<4} "
                    f"win={stats['win_rate'] * 100:4.0f}%  "
                    f"exp={stats['expectancy_r']:+.3f}R"
                )
        return "\n".join(lines)


def _drawdown(equity_curve: list[float]) -> float:
    """Max peak-to-trough drawdown of an equity curve, as a positive %."""
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        dd = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd * 100.0


def _regime_stats(trades: list[BacktestTrade]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        buckets[t.regime].append(t.r_multiple)
    out: dict[str, dict[str, float]] = {}
    for regime, rs in buckets.items():
        wins = sum(1 for r in rs if r > 0)
        out[regime] = {
            "n": float(len(rs)),
            "win_rate": wins / len(rs) if rs else 0.0,
            "expectancy_r": sum(rs) / len(rs) if rs else 0.0,
        }
    return out


def compute_metrics(
    trades: list[BacktestTrade], risk_per_trade_pct: float = 0.5,
) -> PerformanceReport:
    """Aggregate a pooled, time-sorted trade list into a performance report."""
    n = len(trades)
    if n == 0:
        return PerformanceReport(
            n_trades=0, wins=0, losses=0, win_rate=0.0, expectancy_r=0.0,
            std_r=0.0, profit_factor=0.0, total_return_pct=0.0,
            max_drawdown_pct=0.0, sharpe_annualised=0.0, avg_bars_held=0.0,
            span_days=0.0, trades_per_week=0.0,
        )

    rs = [t.r_multiple for t in trades]
    wins = sum(1 for r in rs if r > 0)
    losses = n - wins
    gross_win = sum(r for r in rs if r > 0)
    gross_loss = abs(sum(r for r in rs if r < 0))
    expectancy = sum(rs) / n
    std_r = math.sqrt(sum((r - expectancy) ** 2 for r in rs) / n) if n > 1 else 0.0

    # Compounded equity curve (entry-time order), risking a fixed % at −1R.
    risk_frac = risk_per_trade_pct / 100.0
    equity = 1.0
    curve = [equity]
    for r in rs:
        equity *= (1 + risk_frac * r)
        curve.append(equity)
    total_return = (equity - 1.0) * 100.0
    max_dd = _drawdown(curve)

    # Annualised Sharpe from per-trade R, scaled by trade frequency.
    span_ms = max(trades[-1].exit_ts - trades[0].entry_ts, 1)
    span_days = span_ms / (24 * 60 * 60 * 1000)
    trades_per_year = n / (span_ms / _MS_PER_YEAR)
    sharpe = (
        (expectancy / std_r) * math.sqrt(trades_per_year) if std_r > 0 else 0.0
    )

    by_outcome: dict[str, int] = defaultdict(int)
    for t in trades:
        by_outcome[t.outcome] += 1

    return PerformanceReport(
        n_trades=n,
        wins=wins,
        losses=losses,
        win_rate=wins / n,
        expectancy_r=expectancy,
        std_r=std_r,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else math.inf,
        total_return_pct=total_return,
        max_drawdown_pct=max_dd,
        sharpe_annualised=sharpe,
        avg_bars_held=sum(t.bars_held for t in trades) / n,
        span_days=span_days,
        trades_per_week=n / (span_days / 7) if span_days > 0 else 0.0,
        by_regime=_regime_stats(trades),
        by_outcome=dict(by_outcome),
    )
