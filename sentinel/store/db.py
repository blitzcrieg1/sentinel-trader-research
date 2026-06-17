"""
Sentinel Trader — SQLite Connection Manager.

Provides an async context-manager backed by **aiosqlite** with:
- WAL mode enabled on first connection.
- Automatic schema creation (``CREATE TABLE IF NOT EXISTS``).
- ``data_dir`` created if it doesn't exist.
- Database path sourced from ``get_settings().db_path``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from sentinel.config import get_settings

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    timestamp_utc   TEXT NOT NULL,
    phase           TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id              TEXT PRIMARY KEY,
    pipeline_run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
    symbol          TEXT NOT NULL,
    features_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS ai_decisions (
    id              TEXT PRIMARY KEY,
    pipeline_run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
    symbol          TEXT NOT NULL,
    raw_response    TEXT NOT NULL,
    parsed_json     TEXT,
    decision        TEXT,
    confidence      REAL,
    model_id        TEXT,
    latency_ms      INTEGER,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS risk_verdicts (
    id              TEXT PRIMARY KEY,
    pipeline_run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
    ai_decision_id  TEXT NOT NULL REFERENCES ai_decisions(id),
    verdict         TEXT NOT NULL,
    veto_reason     TEXT,
    computed_size   REAL,
    computed_leverage INTEGER,
    risk_pct        REAL,
    gates_passed    TEXT,
    gates_failed    TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS execution_attempts (
    id              TEXT PRIMARY KEY,
    pipeline_run_id TEXT NOT NULL REFERENCES pipeline_runs(id),
    risk_verdict_id TEXT NOT NULL REFERENCES risk_verdicts(id),
    broker_type     TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    response_json   TEXT,
    status          TEXT NOT NULL,
    error_message   TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    pipeline_run_id TEXT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    purpose         TEXT NOT NULL,
    price           REAL,
    size            REAL NOT NULL,
    status          TEXT NOT NULL,
    fill_price      REAL,
    fill_time       TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS trades (
    id              TEXT PRIMARY KEY,
    pipeline_run_id TEXT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL,
    size            REAL NOT NULL,
    leverage        INTEGER NOT NULL,
    realized_pnl    REAL,
    fees            REAL,
    open_time       TEXT NOT NULL,
    close_time      TEXT,
    close_reason    TEXT,
    status          TEXT NOT NULL,
    funding_paid    REAL DEFAULT 0.0,
    net_pnl         REAL
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    equity_usdt     REAL NOT NULL,
    unrealized_pnl  REAL NOT NULL DEFAULT 0.0,
    snapshot_type   TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    message         TEXT NOT NULL,
    context_json    TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS trading_lessons (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    pnl_usdt REAL NOT NULL,
    lesson_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trading_lessons_symbol ON trading_lessons(symbol);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX IF NOT EXISTS idx_ai_decisions_symbol_created ON ai_decisions(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_equity_snapshots_created ON equity_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_created ON pipeline_runs(created_at);
"""

_schema_initialised = False

async def _ensure_schema(db: aiosqlite.Connection) -> None:
    global _schema_initialised
    if _schema_initialised:
        return
    await db.executescript(_SCHEMA_SQL)
    await _run_migrations(db)
    _schema_initialised = True
    logger.info("database schema initialised")


async def _run_migrations(db: aiosqlite.Connection) -> None:
    """Idempotent column additions for DBs created before a schema change.

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so new
    columns on the live DB are added here. Each ALTER is guarded by a
    PRAGMA check so this is safe to run on every startup.
    """
    cursor = await db.execute("PRAGMA table_info(trades)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "funding_paid" not in columns:
        await db.execute("ALTER TABLE trades ADD COLUMN funding_paid REAL DEFAULT 0.0")
        logger.info("migration: added trades.funding_paid")
    if "net_pnl" not in columns:
        await db.execute("ALTER TABLE trades ADD COLUMN net_pnl REAL")
        logger.info("migration: added trades.net_pnl")
    await db.commit()

@asynccontextmanager
async def get_connection() -> AsyncIterator[aiosqlite.Connection]:
    settings = get_settings()
    data_dir: Path = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(settings.db_path)

    db = await aiosqlite.connect(db_path)
    try:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        # Many short-lived connections write concurrently (8 pipelines +
        # position loop + admin bot): wait for locks instead of failing.
        await db.execute("PRAGMA busy_timeout=5000")
        # WAL + NORMAL is durable to application crashes and far cheaper
        # than FULL on the SD-card-backed SBC this runs on.
        await db.execute("PRAGMA synchronous=NORMAL")
        db.row_factory = aiosqlite.Row
        await _ensure_schema(db)
        yield db
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()

async def init_db() -> None:
    async with get_connection() as db:
        row = await db.execute("SELECT COUNT(*) FROM state")
        count = (await row.fetchone())[0]
        logger.info("database ready — state table has %d rows", count)
