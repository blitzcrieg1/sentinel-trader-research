"""
Sentinel Trader — Paper Broker.

A simulated execution venue implementing the ``Broker`` ABC for Phase 2.

Simulation model:
- **Balance** is persisted in the SQLite ``state`` table (key
  ``paper_equity``, shared with ``core.portfolio``) so virtual equity
  survives restarts. Open positions/orders live in memory but are
  **rehydrated from the persisted ``trades`` + ``orders`` rows on startup**
  via :meth:`PaperBroker.restore`, so a restart no longer force-closes the
  paper book. (Without restore, the reconcile loop would see an empty venue
  and close every open trade as ``manual`` — which silently destroyed all
  natural TP/SL trade outcomes.)
- **Fills** use the live MEXC ticker ``last`` price, adjusted by configured
  slippage *against* the trade direction. Limit entries fill immediately at
  the limit price (paper simplification, logged as such).
- **Fees** (taker, configurable %) are charged on every fill's notional.
- **Trigger orders** (SL/TP) are evaluated lazily on every ``fetch_*`` call
  via mark-to-market: the 60s position loop therefore drives stop/target
  execution at its own cadence, mirroring real venue latency.

All money math is ``Decimal``. The pipeline — not this class — writes the
``execution_attempts`` / ``orders`` / ``trades`` audit rows, so every fill
is persisted exactly once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Final, Literal, cast
from uuid import uuid4

from sentinel.config import Settings, get_settings
from sentinel.data.market import MarketDataClient, MarketDataError
from sentinel.exec.broker import (
    Broker,
    BrokerOrder,
    BrokerPosition,
    OpenPositionRequest,
    OpenPositionResult,
    OrderPurpose,
    Side,
)
from sentinel.store import get_connection, repo

logger = logging.getLogger(__name__)

#: Shared with sentinel.core.portfolio.KEY_PAPER_EQUITY — same persisted balance.
STATE_KEY_BALANCE: Final[str] = "paper_equity"

#: Terminal (filled/cancelled) orders retained in memory; older ones are
#: pruned so a long-running process doesn't grow without bound.
MAX_TERMINAL_ORDERS: Final[int] = 500

_HUNDRED: Final[Decimal] = Decimal("100")
_TP_PURPOSES: Final[tuple[OrderPurpose, ...]] = ("tp1", "tp2", "tp3")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_order_id() -> str:
    return f"paper-{uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Internal mutable state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PaperOrder:
    id: str
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "trigger"]
    purpose: OrderPurpose
    price: Decimal | None
    amount: Decimal
    status: Literal["open", "filled", "cancelled", "expired", "error"]
    fill_price: Decimal | None = None
    fill_time: str | None = None

    def snapshot(self) -> BrokerOrder:
        return BrokerOrder(
            id=self.id, symbol=self.symbol, side=self.side,
            order_type=self.order_type, purpose=self.purpose,
            price=self.price, amount=self.amount, status=self.status,
            fill_price=self.fill_price, fill_time=self.fill_time,
        )


@dataclass(slots=True)
class _PaperPosition:
    symbol: str
    side: Side
    contracts: Decimal
    entry_price: Decimal
    leverage: int
    contract_size: Decimal
    sl_order_id: str
    tp_order_ids: list[str] = field(default_factory=list)

    def direction(self) -> Decimal:
        return Decimal("1") if self.side == "long" else Decimal("-1")


# ---------------------------------------------------------------------------
# Paper broker
# ---------------------------------------------------------------------------


class PaperBroker(Broker):
    """Simulated MEXC swap venue. See module docstring for the model."""

    def __init__(self, market: MarketDataClient, settings: Settings | None = None) -> None:
        self._market = market
        self._settings: Settings = settings or get_settings()
        self._slippage = Decimal(str(self._settings.paper_slippage_pct)) / _HUNDRED
        self._fee = Decimal(str(self._settings.paper_fee_pct)) / _HUNDRED
        self._positions: dict[str, _PaperPosition] = {}
        self._orders: dict[str, _PaperOrder] = {}
        self._last_price: dict[str, Decimal] = {}

    # -- Broker ABC --------------------------------------------------------

    @property
    def name(self) -> str:
        return "paper"

    async def restore(self) -> int:
        """Rehydrate the in-memory book from persisted open trades + orders.

        Called once at startup (markets must already be loaded). Without
        this, a restart leaves ``_positions`` empty and the reconcile loop
        force-closes every open trade as ``manual`` — destroying natural
        TP/SL outcomes and contaminating the equity curve.

        The active stop order's ``size`` is the authoritative remaining
        contract count (it is re-placed at the remaining size on every
        TP1 / breakeven / trailing move), so already-partially-filled
        positions are restored at their correct current size. Already-filled
        TP proceeds are not double-counted: they were realised into the
        persisted balance before the restart.

        Returns the number of positions restored.
        """
        restored = 0
        async with get_connection() as db:
            open_trades = await repo.get_open_trades(db)
            for trade in open_trades:
                orders = await repo.get_orders_for_symbol(
                    db, trade.symbol, statuses=("open",),
                )
                if not orders:
                    logger.warning(
                        "paper restore: open trade %s (%s) has no open orders — "
                        "leaving for reconcile to close",
                        trade.id, trade.symbol,
                    )
                    continue

                sl_order_id = ""
                tp_order_ids: list[str] = []
                remaining: Decimal | None = None
                for o in orders:
                    self._orders[o.id] = _PaperOrder(
                        id=o.id, symbol=o.symbol,
                        side=cast(Literal["buy", "sell"], o.side),
                        order_type=cast(
                            Literal["market", "limit", "trigger"], o.order_type,
                        ),
                        purpose=cast(OrderPurpose, o.purpose),
                        price=Decimal(str(o.price)) if o.price is not None else None,
                        amount=Decimal(str(o.size)),
                        status="open",
                    )
                    if o.purpose in ("sl", "breakeven_sl", "trailing_sl"):
                        sl_order_id = o.id
                        remaining = Decimal(str(o.size))
                    elif o.purpose in ("tp1", "tp2", "tp3"):
                        tp_order_ids.append(o.id)

                if remaining is None:
                    logger.warning(
                        "paper restore: %s has no active stop order — "
                        "restoring at full trade size (verify SL!)",
                        trade.symbol,
                    )
                    remaining = Decimal(str(trade.size))

                contract_size = self._market.get_precision_spec(
                    trade.symbol,
                ).contract_size
                self._positions[trade.symbol] = _PaperPosition(
                    symbol=trade.symbol,
                    side=cast(Side, trade.side),
                    contracts=remaining,
                    entry_price=Decimal(str(trade.entry_price)),
                    leverage=trade.leverage,
                    contract_size=contract_size,
                    sl_order_id=sl_order_id,
                    tp_order_ids=tp_order_ids,
                )
                restored += 1

        if restored:
            logger.info(
                "paper restore: rehydrated %d open position(s) from DB", restored,
            )
        return restored

    async def open_position(self, request: OpenPositionRequest) -> OpenPositionResult:
        try:
            return await self._open_position(request)
        except Exception as exc:  # noqa: BLE001 — rejections are results, not raises
            logger.error("paper open_position failed for %s: %s", request.symbol, exc)
            return OpenPositionResult(
                success=False, error_message=f"{type(exc).__name__}: {exc}",
            )

    async def close_position(self, symbol: str, reason: str) -> BrokerOrder | None:
        position = self._positions.get(symbol)
        if position is None:
            return None
        price = await self._fetch_price(symbol)
        # Closing crosses the spread: slippage against the position.
        exit_price = (
            price * (Decimal("1") - self._slippage) if position.side == "long"
            else price * (Decimal("1") + self._slippage)
        )
        order = _PaperOrder(
            id=_new_order_id(), symbol=symbol,
            side="sell" if position.side == "long" else "buy",
            order_type="market", purpose="sl",
            price=None, amount=position.contracts, status="filled",
            fill_price=exit_price, fill_time=_utcnow_iso(),
        )
        self._orders[order.id] = order
        await self._realize(position, position.contracts, exit_price)
        self._remove_position(position, cancel_all=True)
        logger.info(
            "paper close_position: %s %s @ %s (%s)",
            symbol, position.side, exit_price, reason,
        )
        return order.snapshot()

    async def place_stop_loss(
        self, symbol: str, position_side: Side, contracts: Decimal, trigger_price: Decimal,
        purpose: OrderPurpose = "sl",
    ) -> BrokerOrder:
        order = _PaperOrder(
            id=_new_order_id(), symbol=symbol,
            side="sell" if position_side == "long" else "buy",
            order_type="trigger", purpose=purpose,
            price=trigger_price, amount=contracts, status="open",
        )
        self._orders[order.id] = order
        position = self._positions.get(symbol)
        if position is not None:
            position.sl_order_id = order.id
        logger.info(
            "paper stop placed: %s %s trigger=%s amount=%s purpose=%s",
            symbol, order.side, trigger_price, contracts, purpose,
        )
        return order.snapshot()

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.symbol != symbol or order.status != "open":
            return False
        order.status = "cancelled"
        logger.info("paper order cancelled: %s (%s %s)", order_id, symbol, order.purpose)
        return True

    async def cancel_all_orders(self, symbol: str) -> int:
        count = 0
        for order in self._orders.values():
            if order.symbol == symbol and order.status == "open":
                order.status = "cancelled"
                count += 1
        if count:
            logger.info("paper cancelled %d order(s) on %s", count, symbol)
        return count

    async def fetch_positions(self) -> tuple[BrokerPosition, ...]:
        await self._mark_to_market()
        result: list[BrokerPosition] = []
        for position in self._positions.values():
            last = self._last_price.get(position.symbol, position.entry_price)
            unrealized = (
                (last - position.entry_price) * position.direction()
                * position.contracts * position.contract_size
            )
            result.append(BrokerPosition(
                symbol=position.symbol, side=position.side,
                contracts=position.contracts, entry_price=position.entry_price,
                unrealized_pnl=unrealized, leverage=position.leverage,
            ))
        return tuple(result)

    async def fetch_open_orders(self, symbol: str | None = None) -> tuple[BrokerOrder, ...]:
        await self._mark_to_market()
        return tuple(
            o.snapshot() for o in self._orders.values()
            if o.status == "open" and (symbol is None or o.symbol == symbol)
        )

    async def fetch_order(self, order_id: str, symbol: str) -> BrokerOrder | None:
        await self._mark_to_market()
        order = self._orders.get(order_id)
        if order is None or order.symbol != symbol:
            return None
        return order.snapshot()

    async def fetch_equity(self) -> Decimal:
        await self._mark_to_market()
        balance = await self._get_balance()
        unrealized = Decimal("0")
        for position in self._positions.values():
            last = self._last_price.get(position.symbol, position.entry_price)
            unrealized += (
                (last - position.entry_price) * position.direction()
                * position.contracts * position.contract_size
            )
        return balance + unrealized

    # -- Opening -------------------------------------------------------------

    async def _open_position(self, request: OpenPositionRequest) -> OpenPositionResult:
        if request.symbol in self._positions:
            return OpenPositionResult(
                success=False,
                error_message=f"paper position already open on {request.symbol}",
            )
        if request.contracts <= 0:
            return OpenPositionResult(success=False, error_message="non-positive contracts")

        contract_size = self._market.get_precision_spec(request.symbol).contract_size
        price = await self._fetch_price(request.symbol)

        if request.entry_type == "limit" and request.limit_price is not None:
            fill_price = request.limit_price  # paper simplification: instant fill
        else:
            # Market order: slippage against the direction of the trade.
            fill_price = (
                price * (Decimal("1") + self._slippage) if request.side == "long"
                else price * (Decimal("1") - self._slippage)
            )

        notional = request.contracts * contract_size * fill_price
        entry_fee = notional * self._fee
        balance = await self._get_balance()
        margin = notional / Decimal(request.leverage)
        if margin + entry_fee > balance:
            return OpenPositionResult(
                success=False,
                error_message=(
                    f"insufficient paper balance: need margin {margin:.4f} + fee "
                    f"{entry_fee:.4f}, have {balance:.4f}"
                ),
            )
        await self._set_balance(balance - entry_fee)

        now = _utcnow_iso()
        entry_side: Literal["buy", "sell"] = "buy" if request.side == "long" else "sell"
        exit_side: Literal["buy", "sell"] = "sell" if request.side == "long" else "buy"

        entry_order = _PaperOrder(
            id=_new_order_id(), symbol=request.symbol, side=entry_side,
            order_type=request.entry_type, purpose="entry",
            price=request.limit_price, amount=request.contracts,
            status="filled", fill_price=fill_price, fill_time=now,
        )
        sl_order = _PaperOrder(
            id=_new_order_id(), symbol=request.symbol, side=exit_side,
            order_type="trigger", purpose="sl",
            price=request.stop_loss_price, amount=request.contracts, status="open",
        )
        self._orders[entry_order.id] = entry_order
        self._orders[sl_order.id] = sl_order

        tp_orders: list[_PaperOrder] = []
        amounts = self._split_tp_amounts(request.contracts, len(request.take_profit_prices))
        for i, (tp_price, amount) in enumerate(zip(request.take_profit_prices, amounts)):
            tp = _PaperOrder(
                id=_new_order_id(), symbol=request.symbol, side=exit_side,
                order_type="limit", purpose=_TP_PURPOSES[i],
                price=tp_price, amount=amount, status="open",
            )
            self._orders[tp.id] = tp
            tp_orders.append(tp)

        self._positions[request.symbol] = _PaperPosition(
            symbol=request.symbol, side=request.side,
            contracts=request.contracts, entry_price=fill_price,
            leverage=request.leverage, contract_size=contract_size,
            sl_order_id=sl_order.id, tp_order_ids=[o.id for o in tp_orders],
        )

        logger.info(
            "paper FILL: %s %s %s contracts @ %s (fee=%.6f, sl=%s, tps=%s)",
            request.symbol, request.side, request.contracts, fill_price,
            entry_fee, request.stop_loss_price,
            [str(p) for p in request.take_profit_prices],
        )
        return OpenPositionResult(
            success=True,
            entry_order=entry_order.snapshot(),
            sl_order=sl_order.snapshot(),
            tp_orders=tuple(o.snapshot() for o in tp_orders),
            fill_price=fill_price,
        )

    @staticmethod
    def _split_tp_amounts(contracts: Decimal, n_tps: int) -> list[Decimal]:
        """Split contracts evenly across TPs; the last one takes the remainder."""
        if n_tps <= 1:
            return [contracts]
        per = (contracts / Decimal(n_tps)).quantize(Decimal("0.00000001"))
        amounts = [per] * (n_tps - 1)
        amounts.append(contracts - per * (n_tps - 1))
        return amounts

    # -- Mark-to-market (trigger simulation) ----------------------------------

    async def _mark_to_market(self) -> None:
        """Evaluate SL/TP triggers against live prices for every open position.

        SL is checked before TPs (conservative: assume the adverse move
        happened first within the polling window).
        """
        for symbol in list(self._positions.keys()):
            try:
                price = await self._fetch_price(symbol)
            except MarketDataError as exc:
                logger.warning("paper mark-to-market skipped for %s: %s", symbol, exc)
                continue

            position = self._positions.get(symbol)
            if position is None:
                continue

            sl = self._orders.get(position.sl_order_id)
            if sl is not None and sl.status == "open" and sl.price is not None:
                sl_hit = (
                    price <= sl.price if position.side == "long" else price >= sl.price
                )
                if sl_hit:
                    await self._fill_exit(position, sl, sl.price)
                    continue  # position fully closed

            for tp_id in list(position.tp_order_ids):
                tp = self._orders.get(tp_id)
                if tp is None or tp.status != "open" or tp.price is None:
                    continue
                tp_hit = (
                    price >= tp.price if position.side == "long" else price <= tp.price
                )
                if tp_hit:
                    await self._fill_exit(position, tp, tp.price)
                    if symbol not in self._positions:
                        break  # last TP closed the position

        self._prune_terminal_orders()

    def _prune_terminal_orders(self) -> None:
        """Drop the oldest terminal orders beyond the retention cap."""
        terminal = [oid for oid, o in self._orders.items() if o.status != "open"]
        for oid in terminal[: max(0, len(terminal) - MAX_TERMINAL_ORDERS)]:
            del self._orders[oid]

    async def _fill_exit(
        self, position: _PaperPosition, order: _PaperOrder, exit_price: Decimal,
    ) -> None:
        """Fill an SL/TP order, realize PnL, shrink or close the position."""
        closed = min(order.amount, position.contracts)
        order.status = "filled"
        order.fill_price = exit_price
        order.fill_time = _utcnow_iso()

        await self._realize(position, closed, exit_price)
        position.contracts -= closed

        logger.info(
            "paper trigger FILL: %s %s %s closed=%s @ %s (remaining=%s)",
            position.symbol, position.side, order.purpose, closed,
            exit_price, position.contracts,
        )

        if position.contracts <= 0 or order.purpose in ("sl", "breakeven_sl"):
            self._remove_position(position, cancel_all=True)

    async def _realize(
        self, position: _PaperPosition, contracts: Decimal, exit_price: Decimal,
    ) -> None:
        """Credit realized PnL minus exit fee to the persisted balance."""
        pnl = (
            (exit_price - position.entry_price) * position.direction()
            * contracts * position.contract_size
        )
        fee = exit_price * contracts * position.contract_size * self._fee
        balance = await self._get_balance()
        await self._set_balance(balance + pnl - fee)
        logger.info(
            "paper realized: %s pnl=%.6f fee=%.6f balance=%.4f",
            position.symbol, pnl, fee, balance + pnl - fee,
        )

    def _remove_position(self, position: _PaperPosition, cancel_all: bool) -> None:
        self._positions.pop(position.symbol, None)
        if cancel_all:
            for order in self._orders.values():
                if order.symbol == position.symbol and order.status == "open":
                    order.status = "cancelled"

    # -- Price / balance helpers ------------------------------------------------

    async def _fetch_price(self, symbol: str) -> Decimal:
        ticker = await self._market.fetch_ticker(symbol)
        price = Decimal(str(ticker.last))
        self._last_price[symbol] = price
        return price

    async def _get_balance(self) -> Decimal:
        async with get_connection() as db:
            raw = await repo.get_state(db, STATE_KEY_BALANCE)
        if raw is None:
            initial = Decimal(str(self._settings.paper_starting_equity))
            await self._set_balance(initial)
            return initial
        try:
            value = Decimal(raw)
        except (InvalidOperation, ValueError):
            logger.error("corrupt paper balance %r — resetting to starting equity", raw)
            value = Decimal(str(self._settings.paper_starting_equity))
            await self._set_balance(value)
        return value

    async def _set_balance(self, value: Decimal) -> None:
        async with get_connection() as db:
            await repo.set_state(db, STATE_KEY_BALANCE, f"{value:.8f}")
