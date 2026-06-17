"""
Sentinel Trader — Portfolio State Loader.

Builds the ``PortfolioState`` consumed by the risk engine:

- **Equity** comes from the broker when one is wired (exchange is the
  source of truth); otherwise from the persisted ``paper_equity`` state
  key (Phase 1 / scanner mode).
- **Daily / weekly start equity** anchors are maintained in the ``state``
  table and rolled over automatically on UTC day / ISO-week boundaries.
  They feed the drawdown gates, so they must survive restarts.
- **Open positions** come from the broker when available, otherwise from
  the local ``trades`` table.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Final

import aiosqlite

from sentinel.exec.broker import Broker
from sentinel.risk.engine import OpenPosition, PortfolioState
from sentinel.store import repo

logger = logging.getLogger(__name__)

KEY_PAPER_EQUITY: Final[str] = "paper_equity"
KEY_DAILY_ANCHOR: Final[str] = "equity.daily_start"
KEY_DAILY_ANCHOR_DATE: Final[str] = "equity.daily_start_date"
KEY_WEEKLY_ANCHOR: Final[str] = "equity.weekly_start"
KEY_WEEKLY_ANCHOR_WEEK: Final[str] = "equity.weekly_start_week"

#: Default virtual equity when no broker and no persisted value exists.
DEFAULT_PAPER_EQUITY: Final[Decimal] = Decimal("10000")


def _parse_decimal(raw: str | None, fallback: Decimal) -> Decimal:
    if not raw:
        return fallback
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError):
        logger.error("corrupt decimal in state table: %r — using %s", raw, fallback)
        return fallback
    return value if value.is_finite() and value > 0 else fallback


async def _current_equity(db: aiosqlite.Connection, broker: Broker | None) -> Decimal:
    if broker is not None:
        return await broker.fetch_equity()
    return _parse_decimal(await repo.get_state(db, KEY_PAPER_EQUITY), DEFAULT_PAPER_EQUITY)


async def _rollover_anchor(
    db: aiosqlite.Connection,
    *,
    anchor_key: str,
    period_key: str,
    current_period: str,
    equity: Decimal,
) -> Decimal:
    """Return the persisted anchor, resetting it to ``equity`` on period change."""
    stored_period = await repo.get_state(db, period_key)
    if stored_period != current_period:
        await repo.set_state(db, anchor_key, str(equity))
        await repo.set_state(db, period_key, current_period)
        logger.info(
            "equity anchor %s rolled over (%s -> %s): %s",
            anchor_key, stored_period, current_period, equity,
        )
        return equity
    return _parse_decimal(await repo.get_state(db, anchor_key), equity)


async def _open_positions(
    db: aiosqlite.Connection, broker: Broker | None,
) -> tuple[OpenPosition, ...]:
    if broker is not None:
        broker_positions = await broker.fetch_positions()
        return tuple(
            OpenPosition(
                symbol=p.symbol,
                side=p.side,
                contracts=p.contracts,
                entry_price=p.entry_price,
            )
            for p in broker_positions
            if p.contracts > 0
        )

    trades = await repo.get_open_trades(db)
    return tuple(
        OpenPosition(
            symbol=t.symbol,
            side=t.side,
            contracts=Decimal(str(t.size)),
            entry_price=Decimal(str(t.entry_price)),
        )
        for t in trades
    )


async def load_portfolio_state(
    db: aiosqlite.Connection, broker: Broker | None,
) -> PortfolioState:
    """Assemble the full ``PortfolioState`` for one risk evaluation."""
    equity = await _current_equity(db, broker)

    now = datetime.now(UTC)
    iso = now.isocalendar()
    daily_start = await _rollover_anchor(
        db,
        anchor_key=KEY_DAILY_ANCHOR,
        period_key=KEY_DAILY_ANCHOR_DATE,
        current_period=now.strftime("%Y-%m-%d"),
        equity=equity,
    )
    weekly_start = await _rollover_anchor(
        db,
        anchor_key=KEY_WEEKLY_ANCHOR,
        period_key=KEY_WEEKLY_ANCHOR_WEEK,
        current_period=f"{iso.year}-W{iso.week:02d}",
        equity=equity,
    )

    return PortfolioState(
        equity_usdt=equity,
        daily_start_equity=daily_start,
        weekly_start_equity=weekly_start,
        open_positions=await _open_positions(db, broker),
    )
