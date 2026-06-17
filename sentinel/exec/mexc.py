"""
Sentinel Trader - MEXC V1 Contract Execution Engine.

Implements the Broker interface for MEXC Futures.
Uses aiohttp to interact directly with contract.mexc.com.
"""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import logging
import time
from decimal import Decimal
from typing import Any

import aiohttp

from sentinel.config import Settings, get_settings
from sentinel.exec.broker import (
    Broker,
    BrokerError,
    BrokerOrder,
    BrokerPosition,
    OpenPositionRequest,
    OpenPositionResult,
    Side,
    OrderPurpose,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MEXC API Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://contract.mexc.com"

_TP_PURPOSES: tuple[OrderPurpose, ...] = ("tp1", "tp2", "tp3")

class MexcBroker(Broker):
    """Live execution broker for MEXC V1 Contract API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._cfg = settings or get_settings()
        self._api_key = self._cfg.mexc_api_key
        self._secret_key = self._cfg.mexc_secret_key
        
        if not self._api_key or not self._secret_key:
            logger.warning("MEXC API keys are missing! Live trading will fail.")
            
        self._precision_cache: dict[str, dict[str, Decimal]] = {}
        self._timeout = aiohttp.ClientTimeout(total=self._cfg.exec_timeout_sec)

    @property
    def name(self) -> str:
        return "mexc"

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, payload_str: str = "") -> str:
        msg = self._api_key + timestamp + payload_str
        return hmac.new(
            self._secret_key.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def _request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{BASE_URL}{endpoint}"
        timestamp = str(int(time.time() * 1000))
        
        payload_str = ""
        if payload is not None:
            payload_str = json.dumps(payload, separators=(",", ":"))
            
        signature = self._sign(timestamp, payload_str)
        
        headers = {
            "ApiKey": self._api_key,
            "Request-Time": timestamp,
            "Signature": signature,
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                if method == "GET":
                    async with session.get(url, headers=headers) as resp:
                        data = await resp.json()
                else:
                    async with session.post(url, headers=headers, data=payload_str) as resp:
                        data = await resp.json()
                        
            if data.get("success") is False:
                raise BrokerError(f"MEXC API Error: {data.get('message')}")
                
            return data
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            raise BrokerError(f"MEXC network/parsing error: {exc}") from exc

    # ------------------------------------------------------------------
    # Precision Cache
    # ------------------------------------------------------------------

    async def _ensure_precision(self, symbol: str) -> None:
        """Fetch and cache contract precision details."""
        if symbol in self._precision_cache:
            return
            
        # In CCXT, symbols are BTC/USDT:USDT, in MEXC V1 they are BTC_USDT
        mexc_sym = symbol.replace("/", "_").split(":")[0]
        
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.get(f"{BASE_URL}/api/v1/contract/detail") as resp:
                data = await resp.json()
                
        if data.get("success") is False:
            raise BrokerError(f"Failed to fetch contract details: {data.get('message')}")
            
        for contract in data.get("data", []):
            sym = contract.get("symbol")
            # map back to CCXT format for internal tracking
            internal_sym = sym.replace("_", "/")
            self._precision_cache[internal_sym] = {
                "price_tick": Decimal(str(contract.get("priceTick", "0.0001"))),
                "vol_step": Decimal(str(contract.get("volStep", "1"))),
            }
            
        if symbol not in self._precision_cache:
            # Fallback
            self._precision_cache[symbol] = {"price_tick": Decimal("0.0001"), "vol_step": Decimal("1")}

    def _format_price(self, symbol: str, price: Decimal) -> Decimal:
        tick = self._precision_cache[symbol]["price_tick"]
        # Round to nearest tick
        return (price / tick).quantize(Decimal("1")) * tick

    def _format_amount(self, symbol: str, amount: Decimal) -> Decimal:
        step = self._precision_cache[symbol]["vol_step"]
        # Round down to nearest step
        return (amount / step).quantize(Decimal("1")) * step

    # ------------------------------------------------------------------
    # Account Methods
    # ------------------------------------------------------------------

    async def fetch_equity(self) -> Decimal:
        data = await self._request("GET", "/api/v1/private/account/assets")
        for asset in data.get("data", []):
            if asset.get("currency") == "USDT":
                # 'equity' = balance + unrealized PnL incl. position margin;
                # 'availableBalance' excludes margin and understates equity.
                raw = asset.get("equity")
                if raw is None:
                    raw = asset.get("availableBalance", 0)
                return Decimal(str(raw))
        return Decimal("0")

    async def fetch_positions(self) -> tuple[BrokerPosition, ...]:
        data = await self._request("GET", "/api/v1/private/position/open_positions")
        positions = []
        for pos in data.get("data", []):
            symbol = pos.get("symbol").replace("_", "/")
            side: Side = "long" if pos.get("positionType") == 1 else "short"
            positions.append(BrokerPosition(
                symbol=symbol,
                side=side,
                contracts=Decimal(str(pos.get("holdVol", 0))),
                entry_price=Decimal(str(pos.get("holdAvgPrice", 0))),
                unrealized_pnl=Decimal(str(pos.get("unrealised", 0))),
                leverage=int(pos.get("leverage", 1)),
            ))
        return tuple(positions)

    # ------------------------------------------------------------------
    # Execution Methods
    # ------------------------------------------------------------------

    async def open_position(self, request: OpenPositionRequest) -> OpenPositionResult:
        await self._ensure_precision(request.symbol)
        mexc_sym = request.symbol.replace("/", "_")
        
        contracts = self._format_amount(request.symbol, request.contracts)
        if contracts <= 0:
            return OpenPositionResult(success=False, error_message="Contracts rounded to 0")

        # Side mapping for MEXC V1:
        # 1: Open Long, 2: Close Short, 3: Open Short, 4: Close Long
        side_code = 1 if request.side == "long" else 3
        
        payload = {
            "symbol": mexc_sym,
            "vol": float(contracts),
            "side": side_code,
            "type": 1 if request.entry_type == "limit" else 5,
            "openType": 1, # 1: Isolated margin
            "leverage": request.leverage
        }
        
        if request.entry_type == "limit" and request.limit_price:
            payload["price"] = float(self._format_price(request.symbol, request.limit_price))
            
        # Place entry.
        try:
            data = await self._request("POST", "/api/v1/private/order/submit", payload)
        except BrokerError as exc:
            return OpenPositionResult(success=False, error_message=str(exc))

        entry_order = BrokerOrder(
            id=str(data.get("data")),
            symbol=request.symbol,
            side="buy" if request.side == "long" else "sell",
            order_type=request.entry_type,
            purpose="entry",
            price=request.limit_price,
            amount=contracts,
            status="open",
        )

        # Place SL. If this fails the entry is live but UNPROTECTED — flatten
        # immediately rather than leave a naked position on the venue.
        try:
            sl_order = await self.place_stop_loss(
                symbol=request.symbol,
                position_side=request.side,
                contracts=contracts,
                trigger_price=request.stop_loss_price,
            )
        except BrokerError as exc:
            logger.critical(
                "SL placement failed after entry on %s — emergency flatten: %s",
                request.symbol, exc,
            )
            flatten_note = ""
            try:
                closed = await self.close_position(
                    request.symbol, reason="sl placement failed",
                )
                if closed is None:
                    flatten_note = " (flatten returned no position — VERIFY MANUALLY)"
            except Exception as close_exc:  # noqa: BLE001 — must report, not raise
                flatten_note = f" (flatten FAILED: {close_exc} — NAKED POSITION, intervene now)"
                logger.critical("emergency flatten failed on %s: %s", request.symbol, close_exc)
            return OpenPositionResult(
                success=False,
                error_message=f"SL placement failed; entry flattened: {exc}{flatten_note}",
            )

        # Place TPs (best-effort: SL already protects the position, so a TP
        # failure degrades the exit plan but is not fatal).
        tp_orders: list[BrokerOrder] = []
        amounts = self._split_tp_amounts(contracts, len(request.take_profit_prices))
        for i, (tp_price, amount) in enumerate(
            zip(request.take_profit_prices, amounts)
        ):
            amount = self._format_amount(request.symbol, amount)
            if amount <= 0:
                continue
            try:
                tp_orders.append(await self.place_stop_loss(
                    symbol=request.symbol,
                    position_side=request.side,
                    contracts=amount,
                    trigger_price=tp_price,
                    purpose=_TP_PURPOSES[i],
                ))
            except BrokerError as exc:
                logger.error(
                    "TP%d placement failed for %s at %s: %s (SL remains active)",
                    i + 1, request.symbol, tp_price, exc,
                )

        return OpenPositionResult(
            success=True,
            entry_order=entry_order,
            sl_order=sl_order,
            tp_orders=tuple(tp_orders),
        )

    @staticmethod
    def _split_tp_amounts(contracts: Decimal, n_tps: int) -> list[Decimal]:
        """Split contracts evenly across TPs; the last one takes the remainder."""
        if n_tps <= 0:
            return []
        if n_tps == 1:
            return [contracts]
        per = contracts / Decimal(n_tps)
        amounts = [per] * (n_tps - 1)
        amounts.append(contracts - per * (n_tps - 1))
        return amounts

    async def close_position(self, symbol: str, reason: str) -> BrokerOrder | None:
        await self._ensure_precision(symbol)
        mexc_sym = symbol.replace("/", "_")
        
        # We need to fetch the position to know the size and side
        positions = await self.fetch_positions()
        pos = next((p for p in positions if p.symbol == symbol), None)
        if not pos:
            return None
            
        side_code = 4 if pos.side == "long" else 2 # 4: Close Long, 2: Close Short
        
        payload = {
            "symbol": mexc_sym,
            "vol": float(pos.contracts),
            "side": side_code,
            "type": 5, # Market close
            "openType": 1
        }
        
        try:
            data = await self._request("POST", "/api/v1/private/order/submit", payload)
            order_id = str(data.get("data"))
            
            return BrokerOrder(
                id=order_id,
                symbol=symbol,
                side="sell" if pos.side == "long" else "buy",
                order_type="market",
                purpose="entry", # Reusing purpose type
                price=None,
                amount=pos.contracts,
                status="open",
            )
        except BrokerError:
            return None

    async def place_stop_loss(
        self, symbol: str, position_side: Side, contracts: Decimal, trigger_price: Decimal,
        purpose: OrderPurpose = "sl",
    ) -> BrokerOrder:
        await self._ensure_precision(symbol)
        mexc_sym = symbol.replace("/", "_")
        
        side_code = 4 if position_side == "long" else 2
        trigger = float(self._format_price(symbol, trigger_price))
        
        payload = {
            "symbol": mexc_sym,
            "vol": float(contracts),
            "side": side_code,
            "type": 1, # Stop Market is triggered then becomes market. V1 uses type=1/5, planType etc.
            "triggerPrice": trigger,
            "executePrice": trigger, # Trigger price (could be different for stop-limit)
            "triggerType": 1, # 1: Last price
        }
        
        data = await self._request("POST", "/api/v1/private/planorder/place", payload)
        order_id = str(data.get("data"))
        
        return BrokerOrder(
            id=order_id,
            symbol=symbol,
            side="sell" if position_side == "long" else "buy",
            order_type="trigger",
            purpose=purpose,
            price=trigger_price,
            amount=contracts,
            status="open",
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        payload = {
            "orderIds": [order_id]
        }
        try:
            await self._request("POST", "/api/v1/private/order/cancel", payload)
            return True
        except BrokerError:
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        mexc_sym = symbol.replace("/", "_")
        payload = {
            "symbol": mexc_sym
        }
        try:
            await self._request("POST", "/api/v1/private/order/cancel_all", payload)
            return 1
        except BrokerError:
            return 0

    async def fetch_open_orders(self, symbol: str | None = None) -> tuple[BrokerOrder, ...]:
        # Dummy implementation for now to satisfy interface
        return ()

    async def fetch_order(self, order_id: str, symbol: str) -> BrokerOrder | None:
        # Dummy implementation for now
        return None
