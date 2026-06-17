"""
Sentinel Trader — Carry Book Persistence.

Serialises the paper carry book to JSON with an atomic write, so open hedges
and realised PnL survive a restart (mirroring the directional bot's restart
survival, but self-contained — the carry strategy keeps its own state file
rather than touching the trading DB schema).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from sentinel.carry.executor import CarryBook
from sentinel.carry.position import CarryPosition


def book_to_dict(book: CarryBook) -> dict:
    return {
        "capital_usdt": book.capital_usdt,
        "spot_fee_pct": book.spot_fee_pct,
        "perp_fee_pct": book.perp_fee_pct,
        "slippage_pct": book.slippage_pct,
        "realized_net": book.realized_net,
        "positions": {s: asdict(p) for s, p in book.positions.items()},
        "closed": [asdict(p) for p in book.closed],
    }


def book_from_dict(d: dict) -> CarryBook:
    book = CarryBook(
        capital_usdt=d["capital_usdt"],
        spot_fee_pct=d.get("spot_fee_pct", 0.05),
        perp_fee_pct=d.get("perp_fee_pct", 0.02),
        slippage_pct=d.get("slippage_pct", 0.05),
    )
    book.realized_net = d.get("realized_net", 0.0)
    book.positions = {s: CarryPosition(**pd) for s, pd in d.get("positions", {}).items()}
    book.closed = [CarryPosition(**pd) for pd in d.get("closed", [])]
    return book


def save_book(book: CarryBook, path: str | Path) -> None:
    """Atomically persist the book (write-temp-then-rename, so a crash mid-write
    never corrupts the existing state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(book_to_dict(book), fh)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_book(path: str | Path, *, default_capital: float = 10_000.0) -> CarryBook:
    """Load a persisted book, or return a fresh one with ``default_capital`` if
    no state file exists yet."""
    path = Path(path)
    if not path.exists():
        return CarryBook(capital_usdt=default_capital)
    with open(path, encoding="utf-8") as fh:
        return book_from_dict(json.load(fh))
