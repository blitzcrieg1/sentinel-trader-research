"""
Backtest — offline threshold sweep over a dumped LLM-replay tape.

Loads the raw per-trade tape written by ``llm_replay --no-gates --dump`` and
recomputes the *gate-realistic* expectancy across a range of min-R:R floors,
holding the other geometry gates (RSI extreme, SL-distance band) at their live
config values. One replay → every threshold, with zero extra API calls.

Filtering the tape by ``rr >= floor`` exactly replicates the live min_rr gate:
the gate only admits/rejects the model's proposed trade, it never alters the
SL/TP, so each admitted trade keeps the ``r_multiple`` it actually realised.

Run::

    python -m sentinel.backtest.sweep /tmp/tape_gemini.json
    python -m sentinel.backtest.sweep /tmp/tape_cerebras.json --floors 0,1.0,1.5,2.0,2.5
"""

from __future__ import annotations

import argparse
import json
import sys

from sentinel.config import Settings, get_settings


def _metrics(trades: list[dict]) -> tuple[int, int, float, float, float] | None:
    n = len(trades)
    if n == 0:
        return None
    rs = [t["r_multiple"] for t in trades]
    wins = sum(1 for r in rs if r > 0)
    exp = sum(rs) / n
    gross_win = sum(r for r in rs if r > 0)
    gross_loss = abs(sum(r for r in rs if r < 0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    return n, wins, exp, pf, wins / n * 100.0


def _passes_other_gates(t: dict, s: Settings) -> bool:
    """RSI-extreme and SL-distance gates (everything except min_rr)."""
    rsi = t.get("rsi_1h")
    if rsi is not None:
        if t["side"] == "long" and rsi > s.rsi_overbought_threshold:
            return False
        if t["side"] == "short" and rsi < s.rsi_oversold_threshold:
            return False
    d = t.get("sl_dist_pct")
    if d is not None and (d < s.min_sl_pct or d > s.max_sl_pct):
        return False
    return True


def sweep(tape: list[dict], settings: Settings, floors: list[float]):
    base = [t for t in tape if _passes_other_gates(t, settings)]
    return [(f, _metrics([t for t in base if t["rr"] >= f])) for f in floors], len(base)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sweep min-R:R floors over a dumped LLM tape")
    p.add_argument("tape")
    p.add_argument("--floors", default="0,1.0,1.5,2.0,2.5,3.0")
    args = p.parse_args(argv)

    with open(args.tape, encoding="utf-8") as fh:
        tape = json.load(fh)
    s = get_settings()
    floors = [float(x) for x in args.floors.split(",")]
    rows, base_n = sweep(tape, s, floors)

    print(f"Tape: {args.tape}  ({len(tape)} raw trades, {base_n} pass RSI+SL gates)")
    print(f"Other gates: RSI {s.rsi_oversold_threshold:.0f}/{s.rsi_overbought_threshold:.0f}, "
          f"SL {s.min_sl_pct}-{s.max_sl_pct}%\n")
    print("R:R floor    n      win%    exp(R)     PF")
    print("-" * 44)
    for f, m in rows:
        if m is None:
            print(f"  >= {f:<4.1f}    0      -       -         -")
            continue
        n, _wins, exp, pf, wr = m
        flag = "  <-- best" if exp == max(
            (mm[2] for _, mm in rows if mm), default=-99) else ""
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"  >= {f:<4.1f}    {n:<5}  {wr:4.0f}   {exp:+.3f}    {pf_s}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
