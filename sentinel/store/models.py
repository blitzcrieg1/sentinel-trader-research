"""
Sentinel Trader — Store Models.

Python dataclasses mirroring every table in the SQLite schema
(see spec §9 — Persistence & Audit Trail).

Each model provides:
- `to_dict()`  → plain dict suitable for JSON serialisation / INSERT binding.
- `from_row()` → classmethod that constructs an instance from a sqlite3.Row or tuple.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import uuid4


def _utcnow() -> str:
    """Return the current UTC timestamp as an ISO-8601 string (no microseconds)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid4())


# ── Pipeline Runs ────────────────────────────────────────────────────────────

@dataclass
class PipelineRun:
    """A single invocation of the scan/decision pipeline for one symbol."""
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id", "symbol", "timestamp_utc", "phase", "outcome", "created_at",
    )

    id: str = field(default_factory=_new_id)
    symbol: str = ""
    timestamp_utc: str = ""
    phase: str = ""          # 'scan', 'pre_gate', 'ai', 'risk', 'exec'
    outcome: str = ""        # 'completed', 'skipped', 'error'
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> PipelineRun:
        return cls(**dict(zip(cls.COLUMNS, row)))


# ── Feature Snapshots ────────────────────────────────────────────────────────

@dataclass
class FeatureSnapshot:
    """Exact FeaturePacket JSON sent to the AI model for a pipeline run."""
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id", "pipeline_run_id", "symbol", "features_json", "created_at",
    )

    id: str = field(default_factory=_new_id)
    pipeline_run_id: str = ""
    symbol: str = ""
    features_json: str = ""  # full FeaturePacket JSON
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> FeatureSnapshot:
        return cls(**dict(zip(cls.COLUMNS, row)))

    @property
    def features(self) -> dict[str, Any]:
        return json.loads(self.features_json) if self.features_json else {}


# ── AI Decisions ─────────────────────────────────────────────────────────────

@dataclass
class AiDecision:
    """Raw + parsed AI response for a pipeline run."""
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id", "pipeline_run_id", "symbol", "raw_response", "parsed_json",
        "decision", "confidence", "model_id", "latency_ms",
        "input_tokens", "output_tokens", "created_at",
    )

    id: str = field(default_factory=_new_id)
    pipeline_run_id: str = ""
    symbol: str = ""
    raw_response: str = ""
    parsed_json: str | None = None
    decision: str | None = None     # 'long', 'short', 'no_trade', 'error'
    confidence: float | None = None
    model_id: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> AiDecision:
        return cls(**dict(zip(cls.COLUMNS, row)))


# ── Risk Verdicts ────────────────────────────────────────────────────────────

@dataclass
class RiskVerdict:
    """Deterministic risk-engine verdict for an AI decision."""
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id", "pipeline_run_id", "ai_decision_id", "verdict", "veto_reason",
        "computed_size", "computed_leverage", "risk_pct",
        "gates_passed", "gates_failed", "created_at",
    )

    id: str = field(default_factory=_new_id)
    pipeline_run_id: str = ""
    ai_decision_id: str = ""
    verdict: str = ""            # 'approve', 'veto'
    veto_reason: str | None = None
    computed_size: float | None = None
    computed_leverage: int | None = None
    risk_pct: float | None = None
    gates_passed: str | None = None  # JSON array of gate names
    gates_failed: str | None = None  # JSON array of gate names
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> RiskVerdict:
        return cls(**dict(zip(cls.COLUMNS, row)))

    @property
    def gates_passed_list(self) -> list[str]:
        return json.loads(self.gates_passed) if self.gates_passed else []

    @property
    def gates_failed_list(self) -> list[str]:
        return json.loads(self.gates_failed) if self.gates_failed else []


# ── Execution Attempts ───────────────────────────────────────────────────────

@dataclass
class ExecutionAttempt:
    """One attempt to execute a trade via the broker."""
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id", "pipeline_run_id", "risk_verdict_id", "broker_type",
        "request_json", "response_json", "status", "error_message",
        "created_at",
    )

    id: str = field(default_factory=_new_id)
    pipeline_run_id: str = ""
    risk_verdict_id: str = ""
    broker_type: str = ""        # 'paper', 'mexc'
    request_json: str = ""
    response_json: str | None = None
    status: str = ""             # 'success', 'error', 'timeout'
    error_message: str | None = None
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> ExecutionAttempt:
        return cls(**dict(zip(cls.COLUMNS, row)))


# ── Orders ───────────────────────────────────────────────────────────────────

@dataclass
class Order:
    """A tracked order (entry, TP, SL, breakeven, etc.)."""
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id", "pipeline_run_id", "symbol", "side", "order_type", "purpose",
        "price", "size", "status", "fill_price", "fill_time",
        "created_at", "updated_at",
    )

    id: str = field(default_factory=_new_id)
    pipeline_run_id: str | None = None
    symbol: str = ""
    side: str = ""               # 'buy', 'sell'
    order_type: str = ""         # 'market', 'limit', 'trigger'
    purpose: str = ""            # 'entry', 'tp1', 'tp2', 'tp3', 'sl', 'breakeven_sl'
    price: float | None = None
    size: float = 0.0
    status: str = ""             # 'open', 'filled', 'cancelled', 'expired', 'error'
    fill_price: float | None = None
    fill_time: str | None = None
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> Order:
        return cls(**dict(zip(cls.COLUMNS, row)))


# ── Trades ───────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """A completed (or open) trade with PnL tracking."""
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id", "pipeline_run_id", "symbol", "side", "entry_price",
        "exit_price", "size", "leverage", "realized_pnl", "fees",
        "open_time", "close_time", "close_reason", "status",
        "funding_paid", "net_pnl",
    )

    id: str = field(default_factory=_new_id)
    pipeline_run_id: str | None = None
    symbol: str = ""
    side: str = ""               # 'long', 'short'
    entry_price: float = 0.0
    exit_price: float | None = None
    size: float = 0.0
    leverage: int = 1
    realized_pnl: float | None = None  # gross price PnL (no costs)
    fees: float | None = None          # round-trip taker fees actually incurred
    open_time: str = field(default_factory=_utcnow)
    close_time: str | None = None
    close_reason: str | None = None  # 'tp1', 'tp2', 'tp3', 'sl', 'breakeven', 'manual', 'panic'
    status: str = "open"             # 'open', 'closed'
    funding_paid: float = 0.0        # funding cost while the position was open
    net_pnl: float | None = None     # realized_pnl - fees - funding_paid (true economic result)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> Trade:
        return cls(**dict(zip(cls.COLUMNS, row)))


# ── Equity Snapshots ─────────────────────────────────────────────────────────

@dataclass
class EquitySnapshot:
    """Periodic or event-triggered equity reading for drawdown + curve tracking."""
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id", "equity_usdt", "unrealized_pnl", "snapshot_type", "created_at",
    )

    id: int | None = None        # AUTOINCREMENT
    equity_usdt: float = 0.0
    unrealized_pnl: float = 0.0
    snapshot_type: str = ""      # 'periodic', 'trade_open', 'trade_close', 'daily_reset'
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["id"] is None:
            del d["id"]
        return d

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> EquitySnapshot:
        return cls(**dict(zip(cls.COLUMNS, row)))


# ── Events ───────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """General-purpose event log entry."""
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id", "event_type", "severity", "message", "context_json", "created_at",
    )

    id: int | None = None        # AUTOINCREMENT
    event_type: str = ""         # 'kill_switch', 'halt', 'resume', 'error', 'warning', 'config_change'
    severity: str = "info"       # 'info', 'warning', 'error', 'critical'
    message: str = ""
    context_json: str | None = None
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["id"] is None:
            del d["id"]
        return d

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> Event:
        return cls(**dict(zip(cls.COLUMNS, row)))


# ── State (key/value) ───────────────────────────────────────────────────────

@dataclass
class State:
    """Persisted key/value pair."""
    COLUMNS: ClassVar[tuple[str, ...]] = ("key", "value", "updated_at")

    key: str = ""
    value: str = ""
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> State:
        return cls(**dict(zip(cls.COLUMNS, row)))


@dataclass
class TradingLesson:
    id: str
    symbol: str
    trade_id: str
    pnl_usdt: float
    lesson_text: str
    created_at: str
