"""Quick smoke test for contract module."""
import json

from sentinel.ai.contract import (
    DECISION_JSON_SCHEMA,
    no_trade_decision,
    parse_ai_response,
)

# 1. no_trade factory
d = no_trade_decision("BTC/USDT")
print(f"1. no_trade OK: decision={d.decision}, symbol={d.symbol}, conf={d.confidence}")

# 2. Parse valid long decision
good = json.dumps({
    "schema_version": "1.0",
    "symbol": "ETH/USDT",
    "decision": "long",
    "confidence": 0.75,
    "timeframe_alignment": "aligned",
    "entry": {"type": "limit", "limit_price": 3500.0},
    "stop_loss_price": 3400.0,
    "take_profit_prices": [3700.0, 3900.0],
    "invalidation": "Break below 3400",
    "rationale": "RSI at 55 on 1h, MACD crossing up.",
    "risk_flags": ["high_funding"],
})
parsed = parse_ai_response(good)
assert parsed is not None
print(f"2. Parse long OK: decision={parsed.decision}, conf={parsed.confidence}")

# 3. Bad JSON returns None
bad = parse_ai_response("{bad json}")
assert bad is None
print(f"3. Bad JSON -> None: {bad is None}")

# 4. SL on wrong side returns None (long with SL above entry)
wrong_sl = json.dumps({
    "schema_version": "1.0",
    "symbol": "ETH/USDT",
    "decision": "long",
    "confidence": 0.75,
    "timeframe_alignment": "aligned",
    "entry": {"type": "limit", "limit_price": 3500.0},
    "stop_loss_price": 3600.0,
    "take_profit_prices": [3700.0],
    "invalidation": "test",
    "rationale": "test",
    "risk_flags": [],
})
wrong = parse_ai_response(wrong_sl)
assert wrong is None
print(f"4. Wrong SL side -> None: {wrong is None}")

# 5. Schema has all required fields
schema_props = list(DECISION_JSON_SCHEMA["properties"].keys())
print(f"5. Schema properties: {schema_props}")
assert len(schema_props) == 11

# 6. Test prompts
from sentinel.ai.prompts import SYSTEM_PROMPT, build_messages

msgs = build_messages({"symbol": "BTC/USDT", "price": 100000}, [], [])
assert len(msgs) == 1
assert msgs[0]["role"] == "user"
print(f"6. Messages OK: {len(msgs)} message(s), role={msgs[0]['role']}")
print(f"   System prompt length: {len(SYSTEM_PROMPT)} chars")

# 7. Test AiClient instantiation (without API key, just class structure)
print("\n=== ALL TESTS PASSED ===")
