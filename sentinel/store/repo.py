"""
Sentinel Trader — Repository (async CRUD).
Single point of data access for all tables. Uses parameterised queries exclusively.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

from sentinel.store.models import (
    AiDecision,
    EquitySnapshot,
    Event,
    ExecutionAttempt,
    FeatureSnapshot,
    Order,
    PipelineRun,
    RiskVerdict,
    Trade,
    _utcnow,
)

logger = logging.getLogger(__name__)

def _row_to_tuple(row: aiosqlite.Row) -> tuple[Any, ...]:
    return tuple(row)

async def insert_pipeline_run(db: aiosqlite.Connection, run: PipelineRun) -> str:
    await db.execute(
        "INSERT INTO pipeline_runs (id, symbol, timestamp_utc, phase, outcome, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (run.id, run.symbol, run.timestamp_utc, run.phase, run.outcome, run.created_at),
    )
    await db.commit()
    return run.id

async def update_pipeline_run(
    db: aiosqlite.Connection, run_id: str, phase: str, outcome: str
) -> None:
    await db.execute(
        "UPDATE pipeline_runs SET phase = ?, outcome = ? WHERE id = ?",
        (phase, outcome, run_id),
    )
    await db.commit()

async def insert_feature_snapshot(db: aiosqlite.Connection, snap: FeatureSnapshot) -> str:
    await db.execute(
        "INSERT INTO feature_snapshots (id, pipeline_run_id, symbol, features_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (snap.id, snap.pipeline_run_id, snap.symbol, snap.features_json, snap.created_at),
    )
    await db.commit()
    return snap.id

async def insert_ai_decision(db: aiosqlite.Connection, dec: AiDecision) -> str:
    await db.execute(
        """INSERT INTO ai_decisions 
        (id, pipeline_run_id, symbol, raw_response, parsed_json, decision, confidence, model_id, latency_ms, input_tokens, output_tokens, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (dec.id, dec.pipeline_run_id, dec.symbol, dec.raw_response, dec.parsed_json, dec.decision, dec.confidence, dec.model_id, dec.latency_ms, dec.input_tokens, dec.output_tokens, dec.created_at),
    )
    await db.commit()
    return dec.id

async def get_recent_decisions(db: aiosqlite.Connection, symbol: str, limit: int = 10) -> list[AiDecision]:
    cursor = await db.execute(
        "SELECT * FROM ai_decisions WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
        (symbol, limit),
    )
    return [AiDecision.from_row(_row_to_tuple(r)) for r in await cursor.fetchall()]

async def get_recent_vetoes(db: aiosqlite.Connection, symbol: str, limit: int = 5) -> list[dict]:
    """Return the last N vetoed risk verdicts for a symbol with their AI decision context."""
    cursor = await db.execute(
        """SELECT d.decision, d.confidence, rv.gates_failed, rv.veto_reason, rv.created_at
           FROM risk_verdicts rv
           JOIN ai_decisions d ON rv.ai_decision_id = d.id
           WHERE d.symbol = ? AND rv.verdict = 'vetoed'
           ORDER BY rv.created_at DESC
           LIMIT ?""",
        (symbol, limit),
    )
    rows = await cursor.fetchall()
    return [
        {
            "decision": row[0],
            "confidence": row[1],
            "gate": row[2],
            "reason": row[3],
            "at": row[4],
        }
        for row in rows
    ]

async def insert_risk_verdict(db: aiosqlite.Connection, rv: RiskVerdict) -> str:
    await db.execute(
        """INSERT INTO risk_verdicts 
        (id, pipeline_run_id, ai_decision_id, verdict, veto_reason, computed_size, computed_leverage, risk_pct, gates_passed, gates_failed, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (rv.id, rv.pipeline_run_id, rv.ai_decision_id, rv.verdict, rv.veto_reason, rv.computed_size, rv.computed_leverage, rv.risk_pct, rv.gates_passed, rv.gates_failed, rv.created_at),
    )
    await db.commit()
    return rv.id

async def insert_execution_attempt(db: aiosqlite.Connection, ea: ExecutionAttempt) -> str:
    await db.execute(
        """INSERT INTO execution_attempts 
        (id, pipeline_run_id, risk_verdict_id, broker_type, request_json, response_json, status, error_message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ea.id, ea.pipeline_run_id, ea.risk_verdict_id, ea.broker_type, ea.request_json, ea.response_json, ea.status, ea.error_message, ea.created_at),
    )
    await db.commit()
    return ea.id

async def insert_order(db: aiosqlite.Connection, order: Order) -> str:
    await db.execute(
        """INSERT INTO orders 
        (id, pipeline_run_id, symbol, side, order_type, purpose, price, size, status, fill_price, fill_time, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (order.id, order.pipeline_run_id, order.symbol, order.side, order.order_type, order.purpose, order.price, order.size, order.status, order.fill_price, order.fill_time, order.created_at, order.updated_at),
    )
    await db.commit()
    return order.id

async def insert_trade(db: aiosqlite.Connection, trade: Trade) -> str:
    await db.execute(
        """INSERT INTO trades 
        (id, pipeline_run_id, symbol, side, entry_price, exit_price, size, leverage, realized_pnl, fees, open_time, close_time, close_reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (trade.id, trade.pipeline_run_id, trade.symbol, trade.side, trade.entry_price, trade.exit_price, trade.size, trade.leverage, trade.realized_pnl, trade.fees, trade.open_time, trade.close_time, trade.close_reason, trade.status),
    )
    await db.commit()
    return trade.id

async def get_open_trades(db: aiosqlite.Connection) -> list[Trade]:
    cursor = await db.execute("SELECT * FROM trades WHERE status = 'open' ORDER BY open_time ASC")
    return [Trade.from_row(_row_to_tuple(r)) for r in await cursor.fetchall()]

async def close_trade(
    db: aiosqlite.Connection,
    trade_id: str,
    exit_price: float,
    realized_pnl: float,
    fees: float,
    close_reason: str,
    funding_paid: float = 0.0,
    net_pnl: float | None = None,
) -> None:
    """Close a trade, recording gross PnL plus the true cost breakdown.

    ``realized_pnl`` is the gross price PnL; ``net_pnl`` (when provided) is
    that minus ``fees`` and ``funding_paid`` — the true economic result that
    reconciles with the broker's equity delta.
    """
    await db.execute(
        """UPDATE trades SET status = 'closed', exit_price = ?, realized_pnl = ?,
        fees = ?, funding_paid = ?, net_pnl = ?, close_time = ?, close_reason = ?
        WHERE id = ?""",
        (exit_price, realized_pnl, fees, funding_paid, net_pnl,
         _utcnow(), close_reason, trade_id),
    )
    await db.commit()

async def get_orders_for_symbol(
    db: aiosqlite.Connection, symbol: str, statuses: tuple[str, ...] = ("open",)
) -> list[Order]:
    placeholders = ",".join("?" * len(statuses))
    cursor = await db.execute(
        f"SELECT * FROM orders WHERE symbol = ? AND status IN ({placeholders}) "
        "ORDER BY created_at ASC",
        (symbol, *statuses),
    )
    return [Order.from_row(_row_to_tuple(r)) for r in await cursor.fetchall()]

async def update_order_status(
    db: aiosqlite.Connection,
    order_id: str,
    status: str,
    fill_price: float | None = None,
    fill_time: str | None = None,
) -> None:
    await db.execute(
        """UPDATE orders SET status = ?, fill_price = COALESCE(?, fill_price),
        fill_time = COALESCE(?, fill_time), updated_at = ? WHERE id = ?""",
        (status, fill_price, fill_time, _utcnow(), order_id),
    )
    await db.commit()

async def get_recent_trades(db: aiosqlite.Connection, limit: int = 20) -> list[Trade]:
    cursor = await db.execute("SELECT * FROM trades ORDER BY open_time DESC LIMIT ?", (limit,))
    return [Trade.from_row(_row_to_tuple(r)) for r in await cursor.fetchall()]

async def insert_equity_snapshot(db: aiosqlite.Connection, snap: EquitySnapshot) -> int:
    cursor = await db.execute(
        "INSERT INTO equity_snapshots (equity_usdt, unrealized_pnl, snapshot_type, created_at) VALUES (?, ?, ?, ?)",
        (snap.equity_usdt, snap.unrealized_pnl, snap.snapshot_type, snap.created_at),
    )
    await db.commit()
    return cursor.lastrowid or 0

async def get_equity_history(db: aiosqlite.Connection, hours: int = 24) -> list[EquitySnapshot]:
    cutoff = (datetime.now(UTC) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cursor = await db.execute("SELECT * FROM equity_snapshots WHERE created_at >= ? ORDER BY created_at ASC", (cutoff,))
    return [EquitySnapshot.from_row(_row_to_tuple(r)) for r in await cursor.fetchall()]

async def insert_event(db: aiosqlite.Connection, event: Event) -> int:
    cursor = await db.execute(
        "INSERT INTO events (event_type, severity, message, context_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (event.event_type, event.severity, event.message, event.context_json, event.created_at),
    )
    await db.commit()
    return cursor.lastrowid or 0

async def get_state(db: aiosqlite.Connection, key: str) -> str | None:
    cursor = await db.execute("SELECT value FROM state WHERE key = ?", (key,))
    row = await cursor.fetchone()
    return str(row[0]) if row else None

async def set_state(db: aiosqlite.Connection, key: str, value: str) -> None:
    await db.execute(
        """INSERT INTO state (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
        (key, value, _utcnow()),
    )
    await db.commit()


async def get_trades_since(db: aiosqlite.Connection, since_iso: str) -> list[Trade]:
    cursor = await db.execute(
        "SELECT * FROM trades WHERE open_time >= ? ORDER BY open_time DESC",
        (since_iso,),
    )
    return [Trade.from_row(_row_to_tuple(r)) for r in await cursor.fetchall()]


async def get_trade_stats(db: aiosqlite.Connection, since_iso: str) -> dict[str, Any]:
    cursor = await db.execute(
        """SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) AS losses,
            COALESCE(SUM(realized_pnl), 0.0) AS total_pnl,
            COALESCE(MAX(realized_pnl), 0.0) AS best_pnl,
            COALESCE(MIN(realized_pnl), 0.0) AS worst_pnl
        FROM trades
        WHERE status = 'closed' AND open_time >= ?""",
        (since_iso,),
    )
    row = await cursor.fetchone()
    if row is None or row[0] == 0:
        return {
            "total": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "best_pnl": 0.0, "worst_pnl": 0.0,
            "win_rate": 0.0,
        }
    total, wins, losses, total_pnl, best_pnl, worst_pnl = (
        int(row[0]), int(row[1]), int(row[2]),
        float(row[3]), float(row[4]), float(row[5]),
    )
    return {
        "total": total, "wins": wins, "losses": losses,
        "total_pnl": total_pnl, "best_pnl": best_pnl, "worst_pnl": worst_pnl,
        "win_rate": round(wins / total, 4) if total else 0.0,
    }


async def get_closed_pnl_stats(
    db: aiosqlite.Connection, since_close_time: str | None = None,
) -> dict[str, Any]:
    """Aggregate closed-trade PnL; optional ``since_close_time`` ISO filter on close_time."""
    if since_close_time is None:
        cursor = await db.execute(
            """SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                COALESCE(SUM(realized_pnl), 0.0) AS total_pnl
            FROM trades
            WHERE status = 'closed'""",
        )
    else:
        cursor = await db.execute(
            """SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                COALESCE(SUM(realized_pnl), 0.0) AS total_pnl
            FROM trades
            WHERE status = 'closed' AND close_time >= ?""",
            (since_close_time,),
        )
    row = await cursor.fetchone()
    if row is None or row[0] == 0:
        return {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
    return {
        "total": int(row[0]),
        "wins": int(row[1]),
        "losses": int(row[2]),
        "total_pnl": float(row[3]),
    }


async def get_calibration_samples(
    db: aiosqlite.Connection, clean_only: bool = False,
) -> list[dict[str, Any]]:
    """Recover (confidence → outcome) pairs for confidence calibration.

    Joins each closed trade back to the AI decision that opened it (via the
    shared ``pipeline_run_id``) so the model's stated confidence travels with
    the realised win/loss. No extra logging is needed — the linkage already
    exists in the audit trail.

    Args:
        clean_only: when True, restrict to trades that closed on a real
            stop-loss or take-profit (``close_reason`` in sl/tp1/tp2/tp3).
            Manual/panic closes are excluded because their PnL reflects an
            operator action, not whether the original confidence was right.

    Returns:
        One dict per closed trade with keys: symbol, side, confidence,
        realized_pnl, win (1/0), close_reason. Rows missing a confidence
        (no linked decision) are skipped.
    """
    sql = """
        SELECT t.symbol, t.side, d.confidence,
               COALESCE(t.net_pnl, t.realized_pnl) AS pnl, t.close_reason
        FROM trades t
        JOIN ai_decisions d ON t.pipeline_run_id = d.pipeline_run_id
        WHERE t.status = 'closed'
          AND t.realized_pnl IS NOT NULL
          AND d.confidence IS NOT NULL
    """
    if clean_only:
        sql += " AND t.close_reason IN ('sl', 'tp1', 'tp2', 'tp3')"
    sql += " ORDER BY t.close_time ASC"

    cursor = await db.execute(sql)
    rows = await cursor.fetchall()
    return [
        {
            "symbol": r[0],
            "side": r[1],
            "confidence": float(r[2]),
            "realized_pnl": float(r[3]),
            "win": 1 if float(r[3]) > 0 else 0,
            "close_reason": r[4],
        }
        for r in rows
    ]


async def insert_lesson(db: aiosqlite.Connection, lesson: "TradingLesson") -> str:
    """Persist a new AI trading lesson."""
    import uuid
    if not lesson.id:
        lesson.id = str(uuid.uuid4())
    await db.execute(
        """
        INSERT INTO trading_lessons (id, symbol, trade_id, pnl_usdt, lesson_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (lesson.id, lesson.symbol, lesson.trade_id, lesson.pnl_usdt, lesson.lesson_text, lesson.created_at)
    )
    await db.commit()
    return lesson.id

async def get_recent_lessons(db: aiosqlite.Connection, symbol: str, limit: int = 3) -> list:
    """Fetch the most recent trading lessons for a symbol."""
    from sentinel.store.models import TradingLesson
    rows = await db.execute_fetchall(
        "SELECT id, symbol, trade_id, pnl_usdt, lesson_text, created_at FROM trading_lessons WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
        (symbol, limit)
    )
    return [TradingLesson(*r) for r in rows]
