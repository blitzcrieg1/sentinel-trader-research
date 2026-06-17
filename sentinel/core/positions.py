"""
Sentinel Trader — Position Manager (fast loop).

Runs every ~60 seconds, independently of the scan loop. Responsibilities
(spec §8, Position Management):

1. **Reconcile** — exchange is the source of truth. For every locally-open
   trade, fetch the live position; if the exchange shows it gone, close the
   local trade, log realized PnL, and feed the win/loss counters.
2. **Breakeven after TP1** — once the TP1 order has filled, cancel the
   original SL and place a new stop at the entry price.
3. **Orphan sweep** — cancel any exchange orders attached to symbols that
   no longer have an open position.
4. **Equity snapshots** — persist a periodic equity reading for the curve
   and drawdown anchors.

Fail-safety: each iteration is individually fenced. Any exception is
caught, logged as an ``Event`` row, and the loop continues — one bad tick
never kills position management. With no broker configured (Phase 1), the
loop idles cheaply.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal, cast

import aiosqlite

from sentinel.ai.reflection import run_trade_postmortem
from sentinel.core.pipeline import EXECUTION_LOCK, PipelineContext
from sentinel.core.portfolio import load_portfolio_state
from sentinel.exec.broker import Broker, BrokerOrder, BrokerPosition
from sentinel.store import get_connection, repo
from sentinel.store.models import EquitySnapshot, Event, Order, Trade

logger = logging.getLogger(__name__)

#: Strong references to fire-and-forget tasks (post-mortems) so the event
#: loop cannot garbage-collect them mid-flight.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()

#: Symbols recently closed by reconcile — debounce orphan re-adoption.
_RECENTLY_CLOSED: dict[str, float] = {}
_CLOSE_CONFIRM_DELAY_SEC: float = 0.5
_ORPHAN_DEBOUNCE_SEC: float = 120.0


def _spawn_background(coro, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def _contract_size_for(ctx: PipelineContext, symbol: str) -> Decimal:
    """Exchange contract size for PnL math; conservative 1 if unresolvable."""
    try:
        return ctx.market.get_precision_spec(symbol).contract_size
    except Exception as exc:  # noqa: BLE001 — never let PnL math crash the loop
        logger.warning("cannot resolve contract size for %s (assuming 1): %s", symbol, exc)
        return Decimal("1")


def round_trip_fee(
    entry_price: float, exit_price: float, size: float,
    contract_size: Decimal, fee_pct: float,
) -> float:
    """Taker fees actually paid on entry + exit (rate × notional on each side).

    Matches the paper broker's accounting (it charges ``fee_pct`` of the fill
    notional on every fill), so ``realized_pnl - fees`` reconciles with the
    equity delta. For a live venue this is an estimate until the income
    endpoint is reconciled.
    """
    rate = Decimal(str(fee_pct)) / Decimal("100")
    contracts = Decimal(str(size))
    entry_notional = Decimal(str(entry_price)) * contracts * contract_size
    exit_notional = Decimal(str(exit_price)) * contracts * contract_size
    return float(rate * (entry_notional + exit_notional))


# ---------------------------------------------------------------------------
# Loop entry point
# ---------------------------------------------------------------------------


async def manage_open_positions(ctx: PipelineContext, stop_event: asyncio.Event) -> None:
    """The 60s fast loop. Runs until ``stop_event`` is set. Never raises."""
    interval = float(ctx.cfg.position_loop_interval_sec)
    logger.info("position manager started (interval=%.0fs)", interval)

    if ctx.broker is None:
        logger.info("position manager idle: no broker configured (Phase 1)")

    while not stop_event.is_set():
        if ctx.broker is not None:
            try:
                async with get_connection() as db:
                    await _tick(ctx, db, ctx.broker)
            except Exception as exc:
                logger.exception("position manager tick failed: %s", exc)
                await _log_event_best_effort(
                    "error", "error", f"Position manager tick failed: {exc}",
                    {"exception_type": type(exc).__name__},
                )

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break  # shutdown requested
        except TimeoutError:
            continue

    logger.info("position manager stopped")


# ---------------------------------------------------------------------------
# One tick
# ---------------------------------------------------------------------------


async def _tick(ctx: PipelineContext, db: aiosqlite.Connection, broker: Broker) -> None:
    """One reconciliation pass. Exchange state wins on every mismatch."""
    open_trades = await repo.get_open_trades(db)
    positions = {p.symbol: p for p in await broker.fetch_positions() if p.contracts > 0}
    open_orders = await broker.fetch_open_orders()

    # 0 — adopt orphan positions: a venue position with no local trade row
    # (e.g. a broker call that timed out *after* the fill) would otherwise
    # be completely unmanaged. Exchange is the source of truth — track it.
    await _adopt_orphan_positions(ctx, db, open_trades, positions)

    # 1 + 2 — reconcile each locally-open trade and manage its stops.
    for trade in open_trades:
        position = positions.get(trade.symbol)
        if position is None:
            # Guard against transient empty fetch_positions() (paper broker / network).
            await asyncio.sleep(_CLOSE_CONFIRM_DELAY_SEC)
            retry = {
                p.symbol: p
                for p in await broker.fetch_positions()
                if p.contracts > 0
            }
            position = retry.get(trade.symbol)
        if position is None:
            await _handle_closed_position(ctx, db, broker, trade)
            _RECENTLY_CLOSED[trade.symbol] = time.monotonic()
        else:
            await _manage_breakeven(ctx, db, broker, trade, position)
            await _manage_trailing_stop(ctx, db, broker, trade, position)

    # 3 — orphan sweep: orders whose symbol has neither a live position
    # nor a locally-open trade are stale and consume margin.
    open_trade_symbols = {t.symbol for t in open_trades}
    await _sweep_orphan_orders(db, broker, open_orders, positions, open_trade_symbols)

    # 4 — equity snapshot for the curve.
    equity = await broker.fetch_equity()
    unrealized = sum((p.unrealized_pnl for p in positions.values()), Decimal("0"))
    await repo.insert_equity_snapshot(db, EquitySnapshot(
        equity_usdt=float(equity),
        unrealized_pnl=float(unrealized),
        snapshot_type="periodic",
    ))

    # 5 — continuous drawdown guard: positions bleeding *between* scans must
    # trigger the same daily/weekly halts the risk engine applies at entry.
    await _enforce_drawdown_halts(ctx, db, broker)


async def _adopt_orphan_positions(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    open_trades: list[Trade],
    positions: dict[str, BrokerPosition],
) -> None:
    """Create local trade rows for venue positions we don't know about."""
    local_symbols = {t.symbol for t in open_trades}
    orphans = [p for s, p in positions.items() if s not in local_symbols]
    if not orphans:
        return

    now = time.monotonic()
    for sym, closed_at in list(_RECENTLY_CLOSED.items()):
        if now - closed_at > _ORPHAN_DEBOUNCE_SEC:
            del _RECENTLY_CLOSED[sym]

    # Re-read under the execution lock: a pipeline may have filled the
    # position but not yet committed its trade row when we first looked.
    async with EXECUTION_LOCK:
        local_symbols = {t.symbol for t in await repo.get_open_trades(db)}

    for position in orphans:
        symbol = position.symbol
        if symbol in local_symbols:
            continue
        if now - _RECENTLY_CLOSED.get(symbol, 0.0) < _ORPHAN_DEBOUNCE_SEC:
            logger.debug(
                "skipping orphan adopt for %s — closed by reconcile within debounce window",
                symbol,
            )
            continue
        adopted = Trade(
            symbol=symbol,
            side=position.side,
            entry_price=float(position.entry_price),
            size=float(position.contracts),
            leverage=position.leverage,
            status="open",
        )
        await repo.insert_trade(db, adopted)
        open_trades.append(adopted)
        await repo.insert_event(db, Event(
            event_type="orphan_position_adopted",
            severity="critical",
            message=(
                f"Adopted untracked venue position: {symbol} {position.side} "
                f"{position.contracts} @ {position.entry_price} — verify protective orders"
            ),
            context_json=json.dumps({"trade_id": adopted.id}),
        ))
        logger.critical(
            "orphan position adopted: %s %s %s @ %s — no local trade row existed",
            symbol, position.side, position.contracts, position.entry_price,
        )
        await ctx.notify(
            f"⚠️ ORPHAN POSITION ADOPTED — {symbol}\n"
            f"{position.side} {position.contracts} @ {position.entry_price}\n"
            "No local record existed (timed-out execution or manual trade).\n"
            "Now tracked. CHECK ITS STOP-LOSS MANUALLY."
        )


async def _enforce_drawdown_halts(
    ctx: PipelineContext, db: aiosqlite.Connection, broker: Broker,
) -> None:
    """Engage the kill switch (and flatten) on a daily/weekly drawdown breach."""
    if (await ctx.killswitch.status(db)).halted:
        return

    cfg = ctx.cfg
    portfolio = await load_portfolio_state(db, broker)
    checks = (
        ("daily", portfolio.daily_start_equity, cfg.daily_loss_limit_pct, timedelta(hours=24)),
        ("weekly", portfolio.weekly_start_equity, cfg.weekly_loss_limit_pct, None),
    )
    for kind, anchor, limit_pct, expiry in checks:
        if anchor <= 0:
            continue
        dd_pct = (anchor - portfolio.equity_usdt) / anchor * Decimal("100")
        if dd_pct >= Decimal(str(limit_pct)):
            await ctx.killswitch.engage(
                db,
                f"{kind} drawdown breach (position monitor): {dd_pct:.2f}%",
                context={"drawdown_pct": float(dd_pct)},
                expires_at=datetime.now(UTC) + expiry if expiry else None,
            )
            await ctx.notify(
                f"🛑 KILL SWITCH — {kind} drawdown {dd_pct:.2f}% breached "
                f"limit {limit_pct}%.\nTrading halted, positions flattened."
            )
            return


# ---------------------------------------------------------------------------
# Closed-position reconciliation
# ---------------------------------------------------------------------------


async def _handle_closed_position(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    broker: Broker,
    trade: Trade,
) -> None:
    """The exchange no longer shows this position — close the local trade.

    Exit price is taken from the filled SL/TP order when identifiable,
    otherwise estimated from the current ticker (flagged in the event log).
    """
    exit_price, close_reason = await _determine_exit(ctx, db, broker, trade)

    direction = Decimal("1") if trade.side == "long" else Decimal("-1")
    # trade.size is in *contracts*; PnL needs base-asset units (× contract size).
    contract_size = _contract_size_for(ctx, trade.symbol)
    realized_pnl = float(
        (Decimal(str(exit_price)) - Decimal(str(trade.entry_price)))
        * direction * Decimal(str(trade.size)) * contract_size
    )

    # True cost accounting: record the taker fees actually incurred so net
    # PnL reconciles with the equity delta (the paper broker already debited
    # these from the balance). Funding is 0 for now — sub-cycle trades rarely
    # cross an 8h funding boundary; live funding arrives via reconciliation.
    fees = round_trip_fee(
        float(trade.entry_price), exit_price, float(trade.size),
        contract_size, ctx.cfg.paper_fee_pct,
    )
    funding_paid = 0.0
    net_pnl = realized_pnl - fees - funding_paid

    await repo.close_trade(
        db, trade.id,
        exit_price=exit_price,
        realized_pnl=realized_pnl,
        fees=fees,
        close_reason=close_reason,
        funding_paid=funding_paid,
        net_pnl=net_pnl,
    )

    # Flatten any orders left behind for this symbol (pre-empts the orphan sweep).
    cancelled = await broker.cancel_all_orders(trade.symbol)
    for local_order in await repo.get_orders_for_symbol(db, trade.symbol, statuses=("open",)):
        await repo.update_order_status(db, local_order.id, "cancelled")

    if realized_pnl < 0:
        await ctx.killswitch.record_trade_loss(db, realized_pnl)
    else:
        await ctx.killswitch.record_trade_win(db)

    await repo.insert_equity_snapshot(db, EquitySnapshot(
        equity_usdt=float(await broker.fetch_equity()),
        unrealized_pnl=0.0,
        snapshot_type="trade_close",
    ))
    await repo.insert_event(db, Event(
        event_type="trade_closed",
        severity="info",
        message=(
            f"{trade.symbol} {trade.side} closed ({close_reason}): "
            f"pnl={realized_pnl:.4f} USDT, {cancelled} order(s) swept"
        ),
        context_json=json.dumps({
            "trade_id": trade.id,
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "close_reason": close_reason,
        }),
    ))
    logger.info(
        "trade closed: %s %s gross_pnl=%.4f fees=%.4f net_pnl=%.4f reason=%s",
        trade.symbol, trade.side, realized_pnl, fees, net_pnl, close_reason,
    )

    duration = ""
    if trade.open_time:
        try:
            open_dt = datetime.fromisoformat(trade.open_time.replace("Z", "+00:00"))
            close_dt = datetime.now(UTC)
            delta = close_dt - open_dt
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes = remainder // 60
            duration = f"\nDuration: {hours}h {minutes}m"
        except Exception:
            pass

    pnl_str = f"{realized_pnl:+.4f}" if realized_pnl is not None else "N/A"
    pnl_emoji = "🟢" if (realized_pnl or 0) > 0 else "🔴"
    await ctx.notify(
        f"{pnl_emoji} TRADE CLOSED — {trade.symbol}\n"
        f"Side: {trade.side}\n"
        f"Entry: ${trade.entry_price:,.2f}\n"
        f"Exit: ${exit_price:,.2f}\n"
        f"PnL: {pnl_str} USDT\n"
        f"Reason: {close_reason}{duration}"
    )

    # Trigger post-mortem reflection in the background (reference retained).
    _spawn_background(run_trade_postmortem(ctx.ai, trade.id), name=f"postmortem-{trade.id}")


async def _determine_exit(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    broker: Broker,
    trade: Trade,
) -> tuple[float, str]:
    """Find the exit price and reason from filled protective orders.

    Checks the trade's tracked SL/TP orders against the venue; falls back
    to a ticker estimate when no fill is attributable (manual close, ADL…).
    """
    local_orders = await repo.get_orders_for_symbol(db, trade.symbol, statuses=("open",))
    for local_order in local_orders:
        if local_order.purpose not in ("sl", "breakeven_sl", "tp1", "tp2", "tp3"):
            continue
        venue_order = await broker.fetch_order(local_order.id, trade.symbol)
        if venue_order is None or venue_order.status != "filled":
            continue
        await repo.update_order_status(
            db, local_order.id, "filled",
            fill_price=float(venue_order.fill_price) if venue_order.fill_price else None,
            fill_time=venue_order.fill_time,
        )
        fill = venue_order.fill_price or venue_order.price
        if fill is not None:
            reason = "breakeven" if local_order.purpose == "breakeven_sl" else local_order.purpose
            return float(fill), reason

    # No attributable fill — estimate from the ticker and say so.
    try:
        ticker = await ctx.market.fetch_ticker(trade.symbol)
        return ticker.last, "manual"
    except Exception as exc:
        logger.error("cannot estimate exit price for %s: %s", trade.symbol, exc)
        return trade.entry_price, "manual"


# ---------------------------------------------------------------------------
# Stop replacement (fail-safe)
# ---------------------------------------------------------------------------


async def _place_stop_or_restore(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    broker: Broker,
    trade: Trade,
    *,
    position_side: Literal["long", "short"],
    contracts: Decimal,
    trigger_price: Decimal,
    purpose: str,
    restore_price: Decimal | None,
) -> BrokerOrder | None:
    """Place a replacement stop; on failure try to restore the previous one.

    The cancel-then-place sequence has a window where the position is
    unprotected — if the new placement fails we must not leave it naked.
    Returns the new order, or None when placement failed (restoration is
    best-effort and logged).
    """
    try:
        return await broker.place_stop_loss(
            symbol=trade.symbol,
            position_side=position_side,
            contracts=contracts,
            trigger_price=trigger_price,
            purpose=purpose,  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001 — must fall through to restoration
        logger.critical(
            "failed to place %s for %s at %s: %s — attempting to restore previous SL",
            purpose, trade.symbol, trigger_price, exc,
        )

    restored_msg = "NO PREVIOUS SL PRICE KNOWN — POSITION MAY BE UNPROTECTED"
    if restore_price is not None:
        try:
            restored = await broker.place_stop_loss(
                symbol=trade.symbol,
                position_side=position_side,
                contracts=contracts,
                trigger_price=restore_price,
                purpose="sl",
            )
            await repo.insert_order(db, Order(
                id=restored.id,
                pipeline_run_id=trade.pipeline_run_id,
                symbol=trade.symbol,
                side=restored.side,
                order_type=restored.order_type,
                purpose="sl",
                price=float(restored.price) if restored.price is not None else None,
                size=float(restored.amount),
                status=restored.status,
            ))
            restored_msg = f"previous SL restored at {restore_price}"
        except Exception as restore_exc:  # noqa: BLE001
            restored_msg = f"SL RESTORATION FAILED ({restore_exc}) — POSITION UNPROTECTED"
            logger.critical("SL restoration failed for %s: %s", trade.symbol, restore_exc)

    await repo.insert_event(db, Event(
        event_type="stop_replacement_failed",
        severity="critical",
        message=f"{trade.symbol}: {purpose} placement failed; {restored_msg}",
        context_json=json.dumps({"trade_id": trade.id}),
    ))
    await ctx.notify(
        f"🚨 STOP REPLACEMENT FAILED — {trade.symbol}\n"
        f"Could not place {purpose} at {trigger_price}.\n{restored_msg}"
    )
    return None


# ---------------------------------------------------------------------------
# Breakeven after TP1
# ---------------------------------------------------------------------------


async def _manage_breakeven(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    broker: Broker,
    trade: Trade,
    position: BrokerPosition,
) -> None:
    """Move SL to entry once TP1 has filled (makes the remainder risk-free)."""
    local_orders = await repo.get_orders_for_symbol(
        db, trade.symbol, statuses=("open", "filled"),
    )
    by_purpose: dict[str, list[Order]] = {}
    for order in local_orders:
        by_purpose.setdefault(order.purpose, []).append(order)

    if by_purpose.get("breakeven_sl"):
        return  # already moved

    tp1_orders = by_purpose.get("tp1", [])
    if not tp1_orders:
        return

    # Sync TP1 state from the venue if our local row still says open.
    tp1 = tp1_orders[0]
    if tp1.status != "filled":
        venue_tp1 = await broker.fetch_order(tp1.id, trade.symbol)
        if venue_tp1 is None or venue_tp1.status != "filled":
            return  # TP1 not hit yet
        await repo.update_order_status(
            db, tp1.id, "filled",
            fill_price=float(venue_tp1.fill_price) if venue_tp1.fill_price else None,
            fill_time=venue_tp1.fill_time,
        )

    # TP1 confirmed filled → cancel old SL, place breakeven stop at entry.
    old_sl_price: Decimal | None = None
    for old_sl in by_purpose.get("sl", []):
        if old_sl.status == "open":
            if await broker.cancel_order(old_sl.id, trade.symbol):
                await repo.update_order_status(db, old_sl.id, "cancelled")
                if old_sl.price is not None:
                    old_sl_price = Decimal(str(old_sl.price))

    # trades.side is persisted as 'long'/'short' (written only by the pipeline).
    position_side = cast(Literal["long", "short"], trade.side)
    new_sl_or_none = await _place_stop_or_restore(
        ctx, db, broker, trade,
        position_side=position_side,
        contracts=position.contracts,  # remaining size after TP1, per the venue
        trigger_price=Decimal(str(trade.entry_price)),
        purpose="breakeven_sl",
        restore_price=old_sl_price,
    )
    if new_sl_or_none is None:
        return
    new_sl: BrokerOrder = new_sl_or_none
    await repo.insert_order(db, Order(
        id=new_sl.id,
        pipeline_run_id=trade.pipeline_run_id,
        symbol=trade.symbol,
        side=new_sl.side,
        order_type=new_sl.order_type,
        purpose="breakeven_sl",
        price=float(new_sl.price) if new_sl.price is not None else None,
        size=float(new_sl.amount),
        status=new_sl.status,
    ))
    await repo.insert_event(db, Event(
        event_type="breakeven_moved",
        severity="info",
        message=f"{trade.symbol}: TP1 filled — SL moved to breakeven @ {trade.entry_price}",
        context_json=json.dumps({"trade_id": trade.id, "new_sl_order_id": new_sl.id}),
    ))
    logger.info("breakeven: %s SL moved to entry %.8f", trade.symbol, trade.entry_price)


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------


async def _manage_trailing_stop(
    ctx: PipelineContext,
    db: aiosqlite.Connection,
    broker: Broker,
    trade: Trade,
    position: BrokerPosition,
) -> None:
    """Trail the stop loss behind profit once the activation threshold is hit."""
    if not ctx.cfg.trailing_stop_enabled:
        return

    trail_pct = ctx.cfg.trailing_stop_pct
    activation_pct = ctx.cfg.trailing_stop_activation_pct
    entry = Decimal(str(trade.entry_price))
    is_long = trade.side == "long"

    try:
        ticker = await ctx.market.fetch_ticker(trade.symbol)
        current_price = Decimal(str(ticker.last))
    except Exception as exc:
        logger.warning("trailing stop: cannot fetch ticker for %s: %s", trade.symbol, exc)
        return

    if is_long:
        activation_price = entry * (1 + Decimal(str(activation_pct)) / 100)
        if current_price < activation_price:
            return
        new_sl_price = current_price * (1 - Decimal(str(trail_pct)) / 100)
    else:
        activation_price = entry * (1 - Decimal(str(activation_pct)) / 100)
        if current_price > activation_price:
            return
        new_sl_price = current_price * (1 + Decimal(str(trail_pct)) / 100)

    local_orders = await repo.get_orders_for_symbol(
        db, trade.symbol, statuses=("open",),
    )
    existing_sl: Order | None = None
    for order in local_orders:
        if order.purpose in ("sl", "breakeven_sl", "trailing_sl"):
            existing_sl = order
            break

    if existing_sl is None:
        return

    old_sl_price = Decimal(str(existing_sl.price)) if existing_sl.price is not None else None
    if old_sl_price is not None:
        if is_long and new_sl_price <= old_sl_price:
            return
        if not is_long and new_sl_price >= old_sl_price:
            return

    if await broker.cancel_order(existing_sl.id, trade.symbol):
        await repo.update_order_status(db, existing_sl.id, "cancelled")

    position_side = cast(Literal["long", "short"], trade.side)
    new_sl_or_none = await _place_stop_or_restore(
        ctx, db, broker, trade,
        position_side=position_side,
        contracts=position.contracts,
        trigger_price=new_sl_price,
        purpose="trailing_sl",
        restore_price=old_sl_price,
    )
    if new_sl_or_none is None:
        return
    new_sl: BrokerOrder = new_sl_or_none
    await repo.insert_order(db, Order(
        id=new_sl.id,
        pipeline_run_id=trade.pipeline_run_id,
        symbol=trade.symbol,
        side=new_sl.side,
        order_type=new_sl.order_type,
        purpose="trailing_sl",
        price=float(new_sl.price) if new_sl.price is not None else None,
        size=float(new_sl.amount),
        status=new_sl.status,
    ))
    await repo.insert_event(db, Event(
        event_type="trailing_stop_moved",
        severity="info",
        message=(
            f"{trade.symbol}: trailing SL moved "
            f"{old_sl_price} → {new_sl_price:.8f} (price={current_price})"
        ),
        context_json=json.dumps({
            "trade_id": trade.id,
            "new_sl_order_id": new_sl.id,
            "old_sl_price": str(old_sl_price),
            "new_sl_price": str(new_sl_price),
            "current_price": str(current_price),
        }),
    ))
    logger.info(
        "trailing stop: %s SL moved %s → %.8f (price=%s)",
        trade.symbol, old_sl_price, new_sl_price, current_price,
    )


# ---------------------------------------------------------------------------
# Orphan order sweep
# ---------------------------------------------------------------------------


async def _sweep_orphan_orders(
    db: aiosqlite.Connection,
    broker: Broker,
    open_orders: tuple[BrokerOrder, ...],
    positions: dict[str, BrokerPosition],
    open_trade_symbols: set[str],
) -> None:
    """Cancel venue orders on symbols with no live position and no open trade."""
    orphan_symbols = {
        o.symbol for o in open_orders
        if o.symbol not in positions and o.symbol not in open_trade_symbols
    }
    for symbol in sorted(orphan_symbols):
        cancelled = await broker.cancel_all_orders(symbol)
        for local_order in await repo.get_orders_for_symbol(db, symbol, statuses=("open",)):
            await repo.update_order_status(db, local_order.id, "cancelled")
        await repo.insert_event(db, Event(
            event_type="orphan_sweep",
            severity="warning",
            message=f"Swept {cancelled} orphan order(s) on {symbol} (no position, no open trade)",
        ))
        logger.warning("orphan sweep: cancelled %d order(s) on %s", cancelled, symbol)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _log_event_best_effort(
    event_type: str, severity: str, message: str, context: dict[str, str],
) -> None:
    """Persist an event on a fresh connection; swallow secondary failures."""
    try:
        async with get_connection() as db:
            await repo.insert_event(db, Event(
                event_type=event_type,
                severity=severity,
                message=message,
                context_json=json.dumps(context),
            ))
    except Exception:
        logger.exception("failed to persist position-manager event")
