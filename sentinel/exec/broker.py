"""
Sentinel Trader — Broker Interface (ABC).

The single boundary between the deterministic core and any execution venue.
``PaperBroker`` (Phase 2) and ``MexcBroker`` (Phase 3) implement this
interface; the pipeline and position manager are written against it and
never import CCXT order endpoints directly.

Contract rules:
- All currency / amount fields are ``Decimal``.
- Implementations must be **idempotent-friendly**: callers check for an
  existing position on the symbol before calling ``open_position``.
- Implementations must never raise for ordinary rejections — they return
  ``OpenPositionResult(success=False, ...)``. ``BrokerError`` is reserved
  for transport-level failures the caller may retry or escalate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

Side = Literal["long", "short"]
OrderPurpose = Literal["entry", "tp1", "tp2", "tp3", "sl", "breakeven_sl", "trailing_sl"]


class BrokerError(Exception):
    """Transport-level broker failure (network, auth, exchange rejection)."""


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    """A normalized view of one exchange (or simulated) order."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "purpose": self.purpose,
            "price": str(self.price) if self.price is not None else None,
            "amount": str(self.amount),
            "status": self.status,
            "fill_price": str(self.fill_price) if self.fill_price is not None else None,
            "fill_time": self.fill_time,
        }


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """A normalized view of one open position, as reported by the venue."""

    symbol: str
    side: Side
    contracts: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal
    leverage: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "contracts": str(self.contracts),
            "entry_price": str(self.entry_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "leverage": self.leverage,
        }


@dataclass(frozen=True, slots=True)
class OpenPositionRequest:
    """Everything a broker needs to open a position with protective orders.

    Produced exclusively from an approved ``EngineVerdict`` — never from
    raw AI output.
    """

    symbol: str
    side: Side
    entry_type: Literal["market", "limit"]
    limit_price: Decimal | None
    contracts: Decimal
    leverage: int
    stop_loss_price: Decimal
    take_profit_prices: tuple[Decimal, ...]
    pipeline_run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_type": self.entry_type,
            "limit_price": str(self.limit_price) if self.limit_price is not None else None,
            "contracts": str(self.contracts),
            "leverage": self.leverage,
            "stop_loss_price": str(self.stop_loss_price),
            "take_profit_prices": [str(tp) for tp in self.take_profit_prices],
            "pipeline_run_id": self.pipeline_run_id,
        }


@dataclass(frozen=True, slots=True)
class OpenPositionResult:
    """Outcome of an open-position attempt, including all placed orders."""

    success: bool
    error_message: str | None = None
    entry_order: BrokerOrder | None = None
    sl_order: BrokerOrder | None = None
    tp_orders: tuple[BrokerOrder, ...] = field(default=())
    fill_price: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "error_message": self.error_message,
            "entry_order": self.entry_order.to_dict() if self.entry_order else None,
            "sl_order": self.sl_order.to_dict() if self.sl_order else None,
            "tp_orders": [o.to_dict() for o in self.tp_orders],
            "fill_price": str(self.fill_price) if self.fill_price is not None else None,
        }


class Broker(ABC):
    """Abstract execution venue. See module docstring for the contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier persisted in ``execution_attempts.broker_type``."""

    @abstractmethod
    async def open_position(self, request: OpenPositionRequest) -> OpenPositionResult:
        """Open a position and attach SL/TP orders. Never raises for rejections."""

    @abstractmethod
    async def close_position(self, symbol: str, reason: str) -> BrokerOrder | None:
        """Market-close the position on ``symbol``. None if no position exists."""

    @abstractmethod
    async def place_stop_loss(
        self, symbol: str, position_side: Side, contracts: Decimal, trigger_price: Decimal,
        purpose: OrderPurpose = "sl",
    ) -> BrokerOrder:
        """Place a (new) protective stop for an existing position."""

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel one order. True if cancelled, False if already gone."""

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel every open order on ``symbol``. Returns count cancelled."""

    @abstractmethod
    async def fetch_positions(self) -> tuple[BrokerPosition, ...]:
        """All currently open positions (exchange is the source of truth)."""

    @abstractmethod
    async def fetch_open_orders(self, symbol: str | None = None) -> tuple[BrokerOrder, ...]:
        """All open orders, optionally filtered by symbol."""

    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> BrokerOrder | None:
        """Look up one order by id. None if unknown to the venue."""

    @abstractmethod
    async def fetch_equity(self) -> Decimal:
        """Current account equity in USDT (balance + unrealized PnL)."""
