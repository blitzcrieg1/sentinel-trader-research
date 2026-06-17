"""
Sentinel Trader — Confidence Calibration.

Answers the question the backtests couldn't: with full live fidelity (real
order book / OI), does the model's stated ``confidence`` actually predict
which trades win? This is the running tie-breaker on whether the LLM has
edge — read it as clean SL/TP closes accumulate.

It reuses ``repo.get_calibration_samples`` (which recovers confidence→outcome
pairs by joining each closed trade to its opening decision via the shared
``pipeline_run_id`` — no extra logging needed). ``clean_only`` restricts to
trades that closed on a real stop/target, excluding manual/restart closes
whose PnL reflects an operator action rather than the setup's merit.

Metrics:
- **Reliability table** — for each confidence band, predicted (mean
  confidence) vs actual (win rate). A useful model trends upward: higher
  confidence → higher win rate.
- **Brier score** — mean((confidence − outcome)²). 0.25 = an uninformative
  coin-flip; lower is better, higher means the confidence is anti-predictive.

Until there are enough *clean* closes the verdict is withheld — small-sample
noise is not a verdict.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field

from sentinel.store import get_connection, repo

#: Minimum clean samples before a calibration verdict is meaningful.
MIN_SAMPLES_FOR_VERDICT = 30

#: Confidence band edges.
_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01),
)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    n: int
    clean_only: bool
    wins: int
    win_rate: float
    avg_pnl: float
    brier: float
    bands: list[dict] = field(default_factory=list)  # lo, hi, n, predicted, actual_win, avg_pnl

    @property
    def has_verdict(self) -> bool:
        return self.n >= MIN_SAMPLES_FOR_VERDICT

    def verdict(self) -> str:
        if not self.has_verdict:
            kind = "clean" if self.clean_only else "total"
            return (
                f"INSUFFICIENT DATA — {self.n}/{MIN_SAMPLES_FOR_VERDICT} {kind} closes. "
                f"{MIN_SAMPLES_FOR_VERDICT - self.n} more needed before any conclusion."
            )
        if self.brier > 0.25:
            return (
                "ANTI-PREDICTIVE — confidence does worse than a coin flip "
                f"(Brier {self.brier:.3f} > 0.25). The model's conviction is not trustworthy."
            )
        # Check monotonicity: do higher bands win more than lower ones?
        active = [b for b in self.bands if b["n"] >= 3]
        rising = (
            len(active) >= 2
            and active[-1]["actual_win"] > active[0]["actual_win"]
        )
        if rising and self.brier < 0.25:
            return (
                f"USABLE SIGNAL — Brier {self.brier:.3f} < 0.25 and higher confidence "
                "tracks higher win rate. Confidence carries information."
            )
        return (
            f"WEAK/FLAT — Brier {self.brier:.3f}; confidence does not clearly separate "
            "winners from losers yet. Keep accumulating."
        )

    def format_text(self) -> str:
        kind = "CLEAN SL/TP only" if self.clean_only else "ALL closes (incl. manual)"
        lines = [
            f"=== CONFIDENCE CALIBRATION ({kind}) ===",
            f"Samples:        {self.n}",
        ]
        if self.n == 0:
            lines.append("No closed trades with confidence yet.")
            return "\n".join(lines)
        lines += [
            f"Win rate:       {self.win_rate * 100:.1f}%  ({self.wins}W / {self.n - self.wins}L)",
            f"Avg PnL:        {self.avg_pnl:+.3f} USDT/trade",
            f"Brier score:    {self.brier:.3f}  (0.25 = coin-flip; lower is better)",
            "",
            "Confidence -> actual:",
        ]
        for b in self.bands:
            if b["n"] == 0:
                continue
            lines.append(
                f"  {b['lo']:.2f}-{b['hi']:.2f}  n={b['n']:<3} "
                f"predicted={b['predicted'] * 100:4.0f}%  "
                f"actual_win={b['actual_win'] * 100:4.0f}%  "
                f"avgPnL={b['avg_pnl']:+.2f}"
            )
        lines += ["", f"VERDICT: {self.verdict()}"]
        return "\n".join(lines)


def _compute(samples: list[dict], clean_only: bool) -> CalibrationReport:
    n = len(samples)
    if n == 0:
        return CalibrationReport(0, clean_only, 0, 0.0, 0.0, 0.0, [])

    wins = sum(s["win"] for s in samples)
    avg_pnl = sum(s["realized_pnl"] for s in samples) / n
    brier = sum((s["confidence"] - s["win"]) ** 2 for s in samples) / n

    bands: list[dict] = []
    for lo, hi in _BANDS:
        members = [s for s in samples if lo <= s["confidence"] < hi]
        if members:
            bands.append({
                "lo": lo, "hi": min(hi, 1.0), "n": len(members),
                "predicted": sum(s["confidence"] for s in members) / len(members),
                "actual_win": sum(s["win"] for s in members) / len(members),
                "avg_pnl": sum(s["realized_pnl"] for s in members) / len(members),
            })
        else:
            bands.append({"lo": lo, "hi": min(hi, 1.0), "n": 0, "predicted": 0.0,
                          "actual_win": 0.0, "avg_pnl": 0.0})

    return CalibrationReport(
        n=n, clean_only=clean_only, wins=wins, win_rate=wins / n,
        avg_pnl=avg_pnl, brier=brier, bands=bands,
    )


async def compute_calibration(clean_only: bool = True) -> CalibrationReport:
    """Build a calibration report from the live audit trail."""
    async with get_connection() as db:
        samples = await repo.get_calibration_samples(db, clean_only=clean_only)
    return _compute(samples, clean_only)


async def format_daily_summary() -> str:
    """A compact two-line calibration line for the daily Telegram report."""
    clean = await compute_calibration(clean_only=True)
    if clean.n == 0:
        return "🎯 Calibration: no clean SL/TP closes yet"
    head = (
        f"🎯 Calibration: {clean.n} clean closes, "
        f"{clean.win_rate * 100:.0f}% win, Brier {clean.brier:.2f}"
    )
    if not clean.has_verdict:
        head += f" ({MIN_SAMPLES_FOR_VERDICT - clean.n} more for verdict)"
    return head


def main() -> int:
    parser = argparse.ArgumentParser(description="Confidence calibration report")
    parser.add_argument("--all", action="store_true", help="Include manual/restart closes too")
    args = parser.parse_args()

    async def _run() -> None:
        clean = await compute_calibration(clean_only=True)
        print(clean.format_text())
        if args.all:
            print()
            allr = await compute_calibration(clean_only=False)
            print(allr.format_text())

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
