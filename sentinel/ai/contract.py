"""
Sentinel Trader — AI Decision Contract.

Pydantic models that define the structured output contract between
the Anthropic API and the rest of the system.  The JSON Schema exported
from this module is sent to Anthropic as the native Structured Output
schema; the Pydantic model provides a second layer of validation.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums / Literal unions
# ---------------------------------------------------------------------------

DecisionType = Literal["long", "short", "no_trade"]
TimeframeAlignment = Literal["aligned", "mixed", "conflicting"]
EntryType = Literal["market", "limit"]
RiskFlag = Literal[
    "high_funding",
    "low_volume",
    "wide_spread",
    "high_volatility",
    "near_resistance",
    "near_support",
    "news_unknown",
    "conflicting_signals",
]
SymbolType = str

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class EntrySpec(BaseModel):
    """Specifies the entry order type and optional limit price."""

    type: EntryType = Field(
        ...,
        description="Order type: 'market' for immediate fill, 'limit' for price-specified.",
    )
    limit_price: float | None = Field(
        None,
        description="Required if type is 'limit', null if 'market'.",
    )

    @model_validator(mode="after")
    def limit_price_consistency(self) -> EntrySpec:
        """Ensure limit_price is set for limit orders and null for market orders."""
        if self.type == "limit" and self.limit_price is None:
            raise ValueError("limit_price is required when entry type is 'limit'")
        if self.type == "market" and self.limit_price is not None:
            raise ValueError("limit_price must be null when entry type is 'market'")
        return self


# ---------------------------------------------------------------------------
# Main decision model
# ---------------------------------------------------------------------------


class AiDecision(BaseModel):
    """Structured AI decision — the contract between Claude and the risk engine.

    Every field maps 1:1 to the JSON Schema defined in the spec (section 7).
    Validators enforce basic sanity; the risk engine applies stricter gates.
    """

    schema_version: Literal["1.0"] = Field(
        ...,
        description="Schema version for forward compatibility.",
    )
    symbol: SymbolType = Field(
        ...,
        description="The trading pair this decision applies to.",
    )
    decision: DecisionType = Field(
        ...,
        description="The directional trade decision.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's self-assessed confidence (0.0-1.0).",
    )
    timeframe_alignment: TimeframeAlignment = Field(
        ...,
        description="Whether the 15m, 1h, and 4h timeframes agree on direction.",
    )
    entry: EntrySpec = Field(
        ...,
        description="Entry order specification.",
    )
    stop_loss_price: float = Field(
        ...,
        description="Mandatory stop-loss price.",
    )
    take_profit_prices: list[float] = Field(
        ...,
        max_length=3,
        description="1-3 take-profit levels. First = primary target.",
    )
    invalidation: str = Field(
        ...,
        max_length=200,
        description="One sentence: what would invalidate this trade idea.",
    )
    rationale: str = Field(
        ...,
        max_length=1000,
        description="Max 3 sentences referencing only the provided features.",
    )
    risk_flags: list[RiskFlag] = Field(
        ...,
        description="Detected risk conditions from the feature data.",
    )

    # -- Validators --------------------------------------------------------

    @model_validator(mode="after")
    def validate_trade_direction_prices(self) -> AiDecision:
        """For long/short decisions, validate that SL and TPs are on the correct side."""
        if self.decision == "no_trade":
            return self

        # Basic positive price checks for active trades
        if self.stop_loss_price <= 0:
            raise ValueError(f"stop_loss_price must be positive, got {self.stop_loss_price}")
        if not self.take_profit_prices:
            raise ValueError("take_profit_prices must have at least 1 item")
        for i, tp in enumerate(self.take_profit_prices):
            if tp <= 0:
                raise ValueError(f"take_profit_prices[{i}] must be positive, got {tp}")

        # Determine reference price for side checks
        entry_price = self.entry.limit_price if self.entry.type == "limit" else None
        if entry_price is None:
            # Market order: cannot validate SL/TP direction without a reference
            # price.  The risk engine performs this check with the live price.
            return self

        if self.decision == "long":
            if self.stop_loss_price >= entry_price:
                raise ValueError(
                    f"Long SL ({self.stop_loss_price}) must be below entry ({entry_price})"
                )
            for i, tp in enumerate(self.take_profit_prices):
                if tp <= entry_price:
                    raise ValueError(
                        f"Long TP[{i}] ({tp}) must be above entry ({entry_price})"
                    )
        elif self.decision == "short":
            if self.stop_loss_price <= entry_price:
                raise ValueError(
                    f"Short SL ({self.stop_loss_price}) must be above entry ({entry_price})"
                )
            for i, tp in enumerate(self.take_profit_prices):
                if tp >= entry_price:
                    raise ValueError(
                        f"Short TP[{i}] ({tp}) must be below entry ({entry_price})"
                    )

        return self


# ---------------------------------------------------------------------------
# JSON Schema for Anthropic Native Structured Outputs
# ---------------------------------------------------------------------------

DECISION_JSON_SCHEMA: dict = {
    "type": "OBJECT",
    "required": [
        "schema_version",
        "symbol",
        "decision",
        "confidence",
        "timeframe_alignment",
        "entry",
        "stop_loss_price",
        "take_profit_prices",
        "invalidation",
        "rationale",
        "risk_flags"
    ],
    "properties": {
        "schema_version": {
            "type": "STRING",
            "enum": ["1.0"],
            "description": "Schema version for forward compatibility."
        },
        "symbol": {
            "type": "STRING",
            "description": "The trading pair this decision applies to."
        },
        "decision": {
            "type": "STRING",
            "enum": ["long", "short", "no_trade"],
            "description": "The directional trade decision."
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Model's self-assessed confidence in this decision (0.0-1.0)."
        },
        "timeframe_alignment": {
            "type": "STRING",
            "enum": ["aligned", "mixed", "conflicting"],
            "description": "Whether the 15m, 1h, and 4h timeframes agree on direction."
        },
        "entry": {
            "type": "OBJECT",
            "required": ["type"],
            "properties": {
                "type": {
                    "type": "STRING",
                    "enum": ["market", "limit"]
                },
                "limit_price": {
                    "type": "NUMBER",
                    "nullable": True,
                    "description": "Required if type is 'limit', null if 'market'."
                }
            }
        },
        "stop_loss_price": {
            "type": "NUMBER",
            "description": "Mandatory stop-loss price. Must be below entry for long, above for short."
        },
        "take_profit_prices": {
            "type": "ARRAY",
            "items": {"type": "NUMBER"},
            "description": "1-3 take-profit levels. First = primary target."
        },
        "invalidation": {
            "type": "STRING",
            "description": "One sentence: what would invalidate this trade idea."
        },
        "rationale": {
            "type": "STRING",
            "description": "Max 3 sentences referencing only the provided features."
        },
        "risk_flags": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": [
                    "high_funding",
                    "low_volume",
                    "wide_spread",
                    "high_volatility",
                    "near_resistance",
                    "near_support",
                    "news_unknown",
                    "conflicting_signals"
                ]
            },
            "description": "Detected risk conditions from the feature data."
        }
    }
}


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------


def parse_ai_response(raw_json: str) -> AiDecision | None:
    """Parse raw JSON text into a validated ``AiDecision``.

    Returns ``None`` (and logs the error) on *any* failure: malformed JSON,
    missing fields, constraint violations, etc.  The caller is responsible
    for incrementing the malformed-response counter.
    """
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("AI response is not valid JSON: %s", exc)
        return None

    try:
        return AiDecision.model_validate(data)
    except Exception as exc:
        repaired = _try_repair_partial_no_trade(data)
        if repaired is not None:
            logger.warning("Repaired partial no_trade AI response for %s", repaired.symbol)
            return repaired
        logger.error("AI response failed Pydantic validation: %s", exc)
        return None


def _try_repair_partial_no_trade(data: dict) -> AiDecision | None:
    """NVIDIA NIM sometimes omits placeholder fields on no_trade — fill them in."""
    if data.get("decision") != "no_trade":
        return None
    symbol = data.get("symbol")
    if not symbol:
        return None
    confidence = data.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    alignment = data.get("timeframe_alignment", "mixed")
    if alignment not in ("aligned", "mixed", "conflicting"):
        alignment = "mixed"
    flags = data.get("risk_flags")
    if not isinstance(flags, list):
        flags = []
    return no_trade_decision(
        symbol=symbol,
        rationale=str(data.get("rationale") or "No clear setup identified.")[:1000],
        invalidation=str(data.get("invalidation") or "N/A")[:200],
    ).model_copy(
        update={
            "confidence": max(0.0, min(1.0, confidence)),
            "timeframe_alignment": alignment,
            "risk_flags": flags,
        },
    )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def no_trade_decision(
    symbol: SymbolType,
    rationale: str = "No clear setup identified.",
    invalidation: str = "N/A",
) -> AiDecision:
    """Create a valid ``no_trade`` decision for the given symbol.

    Useful as a safe fallback when the AI response is malformed, the budget
    is exhausted, or an upstream error prevents a real call.
    """
    return AiDecision(
        schema_version="1.0",
        symbol=symbol,
        decision="no_trade",
        confidence=0.0,
        timeframe_alignment="mixed",
        entry=EntrySpec(type="market", limit_price=None),
        stop_loss_price=1.0,  # Placeholder: never acted upon for no_trade
        take_profit_prices=[1.0],  # Placeholder: never acted upon for no_trade
        invalidation=invalidation,
        rationale=rationale,
        risk_flags=[],
    )
