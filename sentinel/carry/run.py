"""
Sentinel Trader — Carry Strategy Runner (paper).

The autonomous loop for the delta-neutral funding-carry strategy:
  - every 8h, just after funding settles (00/08/16 UTC), accrue funding and
    run hysteresis exits;
  - once a day, re-scan and rebalance the basket;
  - persist the book after every change (restart survival).

Scheduling decisions are pure functions (unit-tested); the loop itself is thin
network/time-bound glue. Runs independently of the directional bot.

    python -m sentinel.carry.run --capital 10000 --state data/carry_book.json
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime, timedelta

from sentinel.carry.manager import CarryManager
from sentinel.carry.notify import format_carry_report, send_telegram
from sentinel.carry.persistence import load_book, save_book
from sentinel.carry.prices import fetch_market
from sentinel.carry.scanner import scan

logger = logging.getLogger(__name__)

_FUNDING_HOURS = (0, 8, 16)  # UTC settlement hours


def seconds_until_next_settlement(now: datetime, *, buffer_min: int = 5) -> float:
    """Seconds from ``now`` until just after the next 8h funding settlement
    (00:00/08:00/16:00 UTC + ``buffer_min``)."""
    candidates = []
    for day_offset in (0, 1):
        base = (now + timedelta(days=day_offset)).replace(minute=buffer_min, second=0, microsecond=0)
        candidates.extend(base.replace(hour=h) for h in _FUNDING_HOURS)
    nxt = min(c for c in candidates if c > now)
    return (nxt - now).total_seconds()


def should_rebalance(now: datetime, last_rebalance: datetime | None, *, interval_h: float = 24.0) -> bool:
    """True if a basket rebalance is due (none yet, or ``interval_h`` elapsed)."""
    return last_rebalance is None or (now - last_rebalance) >= timedelta(hours=interval_h)


def _run(args) -> int:
    book = load_book(args.state, default_capital=args.capital)
    mgr = CarryManager(
        book, min_consistency=args.min_consistency, min_carry_yr=args.min_carry,
        min_spot_vol_usdt=args.min_spot_vol, max_names=args.max_names,
        max_per_name_frac=args.max_per_name, deploy_frac=args.deploy_frac,
    )
    logger.info("carry runner started: capital=%.0f state=%s open=%d",
                book.capital_usdt, args.state, len(book.positions))
    _report(book, header="CARRY — runner started")

    last_rebalance: datetime | None = None
    while True:
        now = datetime.now(UTC)
        if should_rebalance(now, last_rebalance, interval_h=args.rebalance_h):
            logger.info("carry: scanning + rebalancing basket…")
            stats = scan(min_vol_usdt=args.scan_min_vol)
            plan = mgr.rebalance(stats)
            last_rebalance = now
            save_book(book, args.state)
            logger.info("carry: rebalanced (opens=%d closes=%d)", len(plan.opens), len(plan.closes))
            if not plan.is_noop:
                _report(book, header="CARRY — basket rebalanced")

        wait = seconds_until_next_settlement(datetime.now(UTC))
        logger.info("carry: sleeping %.0fs until next settlement", wait)
        time.sleep(wait)

        collected = mgr.settle()
        save_book(book, args.state)
        logger.info("carry: settled funding=%.4f open=%d realized_net=%.4f",
                    collected, len(book.positions), book.realized_net)
        _report(book, header="CARRY — funding settled")


def _report(book, *, header: str) -> None:
    """Send a Telegram snapshot of the book (best-effort)."""
    market = fetch_market(list(book.positions)) if book.positions else {}
    spot = {s: m["spot"] for s, m in market.items()}
    perp = {s: m["perp"] for s, m in market.items()}
    send_telegram(format_carry_report(book, spot, perp, header=header))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the paper funding-carry strategy")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--state", default="data/carry_book.json")
    p.add_argument("--min-carry", type=float, default=10.0)
    p.add_argument("--min-consistency", type=float, default=85.0)
    p.add_argument("--min-spot-vol", type=float, default=2_000_000.0)
    p.add_argument("--max-names", type=int, default=8)
    p.add_argument("--max-per-name", type=float, default=0.30)
    p.add_argument("--deploy-frac", type=float, default=1.0)
    p.add_argument("--rebalance-h", type=float, default=24.0)
    p.add_argument("--scan-min-vol", type=float, default=3_000_000.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return _run(args)
    except KeyboardInterrupt:
        logger.info("carry runner stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
