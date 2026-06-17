"""
Sentinel Trader — AI package.

Re-exports the key classes and utilities so downstream code can do::

    from sentinel.ai import AiClient, AiDecision, parse_ai_response
"""

from __future__ import annotations

from sentinel.ai.client import AiClient
from sentinel.ai.contract import (
    DECISION_JSON_SCHEMA,
    AiDecision,
    EntrySpec,
    no_trade_decision,
    parse_ai_response,
)
from sentinel.ai.prompts import SYSTEM_PROMPT, build_user_message

__all__ = [
    "DECISION_JSON_SCHEMA",
    "SYSTEM_PROMPT",
    "AiClient",
    "AiDecision",
    "EntrySpec",
    "build_user_message",
    "no_trade_decision",
    "parse_ai_response",
]
