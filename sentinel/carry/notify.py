"""
Sentinel Trader — Carry Telegram Reporting.

A minimal, self-contained Telegram notifier for the carry runner (separate
from the directional bot's admin bot). ``format_carry_report`` is pure and
tested; ``send_telegram`` is best-effort and never raises into the loop.
Credentials come from the same env the directional bot uses
(TELEGRAM_BOT_TOKEN / TELEGRAM_ADMIN_CHAT_ID).
"""

from __future__ import annotations

import logging
import os
import urllib.parse
import urllib.request

from sentinel.carry.executor import CarryBook

logger = logging.getLogger(__name__)


def format_carry_report(
    book: CarryBook, spot_prices: dict[str, float], perp_prices: dict[str, float],
    *, header: str = "CARRY REPORT",
) -> str:
    """Human-readable snapshot of the carry book."""
    equity = book.equity(spot_prices, perp_prices)
    pnl = equity - book.capital_usdt
    funding = book.total_funding_accrued()
    lines = [
        f"🪙 {header}",
        f"💰 Equity: ${equity:,.2f}  ({'+' if pnl >= 0 else ''}{pnl:,.2f})",
        f"📊 Open: {len(book.positions)} positions, ${book.deployed_notional():,.0f} deployed",
    ]
    for sym, p in sorted(book.positions.items()):
        lines.append(f"   {sym}: funding +${p.accrued_funding:,.2f}")
    lines.append(f"💵 Funding accrued: ${funding:,.2f}")
    lines.append(f"📈 Realized net: ${book.realized_net:,.2f}")
    if book.closed:
        lines.append(f"🔁 Closed to date: {len(book.closed)}")
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    """POST a message to the admin chat. Returns True on success; never raises
    (a notification failure must not break the strategy loop)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "")
    if len(token) < 20 or not chat:
        logger.debug("carry telegram disabled (no token/chat)")
        return False
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data,
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — fixed host
            return resp.status == 200
    except Exception:  # noqa: BLE001 — best-effort
        logger.warning("carry telegram send failed")
        return False
