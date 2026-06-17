"""
Sentinel Trader — Position Sizing.

Pure, deterministic sizing math. **All arithmetic is ``Decimal``** — binary
floats never touch a currency amount in this module. Inputs arriving as
floats are converted through ``str()`` to avoid amplifying float artefacts.

Core formula (spec §8):

    risk_usdt     = equity × risk_pct
    raw_amount    = risk_usdt / |entry − stop_loss|          (base-asset units)
    contracts     = floor(raw_amount / contract_size / amount_step) × amount_step

Rounding is always **down** (``ROUND_DOWN``): quantization may only ever
*reduce* risk, never increase it. After quantization, actual risk and
notional are recomputed from the final contract count, and every cap
(leverage, per-symbol notional, minimum notional) is re-checked against
those final values — not the pre-rounding intent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any

from sentinel.data.market import PrecisionSpec

logger = logging.getLogger(__name__)

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")


class SizingError(Exception):
    """Raised when inputs are unusable (not when the result is merely too small)."""


def to_decimal(value: Any, field_name: str) -> Decimal:
    """Strict conversion to a finite ``Decimal`` via ``str()``. Raises ``SizingError``."""
    if value is None:
        raise SizingError(f"sizing input '{field_name}' is None")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise SizingError(f"sizing input '{field_name}' is not numeric: {value!r}") from exc
    if not result.is_finite():
        raise SizingError(f"sizing input '{field_name}' is not finite: {value!r}")
    return result


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Outcome of a sizing computation.

    ``ok=False`` means the trade is *valid but unsizeable* (e.g. below the
    minimum notional) — the engine turns this into a veto with ``reason``.
    """

    ok: bool
    reason: str | None
    contracts: Decimal            # final exchange-quantized contract count
    amount_base: Decimal          # contracts × contract_size (base-asset units)
    notional_usdt: Decimal        # amount_base × entry price
    risk_usdt: Decimal            # actual USDT at risk if SL is hit (post-quantization)
    actual_risk_pct: Decimal      # risk_usdt / equity × 100
    leverage: int                 # minimal integer leverage needed, ≤ cap
    margin_usdt: Decimal          # notional / leverage

    @classmethod
    def rejected(cls, reason: str) -> SizingResult:
        return cls(
            ok=False, reason=reason,
            contracts=_ZERO, amount_base=_ZERO, notional_usdt=_ZERO,
            risk_usdt=_ZERO, actual_risk_pct=_ZERO, leverage=1, margin_usdt=_ZERO,
        )


def quantize_amount(amount: Decimal, step: Decimal) -> Decimal:
    """Floor ``amount`` to a multiple of ``step`` (CCXT ``amount_to_precision``
    semantics with TRUNCATE mode). Never rounds up."""
    if step <= 0:
        raise SizingError(f"invalid amount step: {step}")
    return (amount / step).to_integral_value(rounding=ROUND_DOWN) * step


def compute_position_size(
    *,
    equity_usdt: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    risk_per_trade_pct: Decimal,
    max_leverage: int,
    max_notional_usdt: Decimal,
    min_notional_usdt: Decimal,
    precision: PrecisionSpec,
) -> SizingResult:
    """Compute the deterministic position size for one trade.

    Args:
        equity_usdt: Current account equity in USDT.
        entry_price: Validated entry price (limit price or current market).
        stop_loss_price: Validated stop-loss (side-checked by the engine).
        risk_per_trade_pct: Max risk as percent of equity (e.g. ``0.5``).
        max_leverage: Hard leverage cap (AI can never influence this).
        max_notional_usdt: Per-symbol notional ceiling.
        min_notional_usdt: Below this notional, the trade is skipped.
        precision: Exchange amount step / minimums / contract size.

    Returns:
        ``SizingResult`` — ``ok=False`` with a reason when unsizeable.

    Raises:
        SizingError: On structurally invalid inputs (zero/negative prices,
            SL equal to entry, non-positive equity). These indicate upstream
            bugs, not market conditions.
    """
    if equity_usdt <= 0:
        raise SizingError(f"equity must be positive, got {equity_usdt}")
    if entry_price <= 0 or stop_loss_price <= 0:
        raise SizingError(
            f"prices must be positive: entry={entry_price} sl={stop_loss_price}"
        )
    if risk_per_trade_pct <= 0:
        raise SizingError(f"risk_per_trade_pct must be positive, got {risk_per_trade_pct}")
    if max_leverage < 1:
        raise SizingError(f"max_leverage must be >= 1, got {max_leverage}")

    sl_distance = abs(entry_price - stop_loss_price)
    if sl_distance == 0:
        raise SizingError("stop-loss equals entry price — zero SL distance")

    # ── Risk budget → raw base-asset amount ─────────────────────────────
    risk_budget_usdt = equity_usdt * risk_per_trade_pct / _HUNDRED
    raw_amount_base = risk_budget_usdt / sl_distance

    # ── Cap notional: per-symbol ceiling AND leverage ceiling ───────────
    # The position can never require more margin than full equity at max
    # leverage, and never exceed the configured per-symbol notional.
    notional_cap = min(max_notional_usdt, equity_usdt * Decimal(max_leverage))
    raw_notional = raw_amount_base * entry_price
    if raw_notional > notional_cap:
        raw_amount_base = notional_cap / entry_price
        logger.info(
            "sizing capped by notional: raw=%s cap=%s", raw_notional, notional_cap,
        )

    # ── Quantize to exchange precision (floor — risk only shrinks) ──────
    if precision.contract_size <= 0:
        raise SizingError(f"invalid contract size: {precision.contract_size}")
    raw_contracts = raw_amount_base / precision.contract_size
    contracts = quantize_amount(raw_contracts, precision.amount_step)

    if contracts <= 0:
        return SizingResult.rejected(
            f"size quantized to zero (raw_contracts={raw_contracts:.10f}, "
            f"step={precision.amount_step})"
        )
    if precision.min_amount is not None and contracts < precision.min_amount:
        return SizingResult.rejected(
            f"contracts {contracts} below exchange minimum {precision.min_amount}"
        )

    # ── Recompute everything from the FINAL quantized size ──────────────
    amount_base = contracts * precision.contract_size
    notional_usdt = amount_base * entry_price
    risk_usdt = amount_base * sl_distance
    actual_risk_pct = risk_usdt / equity_usdt * _HUNDRED

    min_notional = max(
        min_notional_usdt,
        precision.min_notional if precision.min_notional is not None else _ZERO,
    )
    if notional_usdt < min_notional:
        return SizingResult.rejected(
            f"notional {notional_usdt:.4f} USDT below minimum {min_notional} USDT"
        )

    # Defensive invariant: quantization floors, so this can never trip —
    # unless the formula above is edited incorrectly. Fail loudly if so.
    if actual_risk_pct > risk_per_trade_pct * Decimal("1.0001"):
        raise SizingError(
            f"post-quantization risk {actual_risk_pct}% exceeds budget "
            f"{risk_per_trade_pct}% — sizing invariant violated"
        )

    # ── Minimal integer leverage that covers the notional with equity ───
    # ceil(notional / equity), clamped to [1, max_leverage].
    leverage = int(-(-notional_usdt // equity_usdt)) if notional_usdt > equity_usdt else 1
    leverage = max(1, min(leverage, max_leverage))
    margin_usdt = notional_usdt / Decimal(leverage)
    if margin_usdt > equity_usdt:
        return SizingResult.rejected(
            f"required margin {margin_usdt:.4f} exceeds equity {equity_usdt:.4f} "
            f"even at {max_leverage}x leverage"
        )

    result = SizingResult(
        ok=True,
        reason=None,
        contracts=contracts,
        amount_base=amount_base,
        notional_usdt=notional_usdt,
        risk_usdt=risk_usdt,
        actual_risk_pct=actual_risk_pct,
        leverage=leverage,
        margin_usdt=margin_usdt,
    )
    logger.info(
        "sizing: contracts=%s notional=%.4f risk=%.4f (%.4f%%) lev=%dx margin=%.4f",
        contracts, notional_usdt, risk_usdt, actual_risk_pct, leverage, margin_usdt,
    )
    return result
