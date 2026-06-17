from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

from sentinel.ai.client import AiClient
from sentinel.store import get_connection, repo
from sentinel.store.models import TradingLesson

logger = logging.getLogger(__name__)

async def run_veto_postmortem(
    ai_client: AiClient,
    symbol: str,
    gate: str,
    veto_reason: str,
    decision: str,
    confidence: float,
    features_json: str | None = None,
) -> None:
    """Extract a trading lesson from a risk-engine veto and persist it."""
    try:
        feat_block = f"\nMARKET FEATURES:\n{features_json}" if features_json else ""
        prompt = (
            f"You are a quantitative trading analyst reviewing a rejected trade signal.\n"
            f"The risk engine's rules are fixed — your job is to understand what needs to change\n"
            f"and extract ONE concise rule to avoid this rejection in the future.\n\n"
            f"VETOED SIGNAL:\n"
            f"Symbol: {symbol}\n"
            f"Direction: {decision}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Rejected by gate: {gate}\n"
            f"Full reason: {veto_reason}\n"
            f"{feat_block}\n\n"
            f"What specific adjustment to SL placement, TP targeting, or entry selection\n"
            f"would have made this signal pass the '{gate}' gate?\n\n"
            f"Reply ONLY with the extracted lesson (max 2 sentences). Start with 'LESSON: '."
        )

        lesson_text_raw = await ai_client.request_text(prompt, temperature=0.2)
        if not lesson_text_raw:
            logger.warning("veto postmortem got no response for %s gate=%s", symbol, gate)
            return

        lesson_text = lesson_text_raw
        if lesson_text.upper().startswith("LESSON:"):
            lesson_text = lesson_text[7:].strip()

        lesson = TradingLesson(
            id=str(uuid.uuid4()),
            symbol=symbol,
            trade_id=f"veto:{gate}",
            pnl_usdt=0.0,
            lesson_text=lesson_text,
            created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        async with get_connection() as db:
            await repo.insert_lesson(db, lesson)

        logger.info("veto lesson for %s [%s]: %s", symbol, gate, lesson_text)

    except Exception as exc:
        logger.error("veto postmortem failed for %s gate=%s: %s", symbol, gate, exc)


async def run_trade_postmortem(ai_client: AiClient, trade_id: str) -> None:
    """Reflect on a closed trade and extract a learning rule."""
    try:
        async with get_connection() as db:
            # 1. Fetch trade details
            rows = await db.execute_fetchall(
                "SELECT symbol, side, entry_price, exit_price, size, realized_pnl, close_reason, open_time, close_time, pipeline_run_id FROM trades WHERE id = ?",
                (trade_id,)
            )
            if not rows:
                return
            t = rows[0]
            symbol, side, entry, exit_px, size, pnl, reason, open_t, close_t, run_id = t

            # 2. Fetch the AI decision that opened it
            ai_rows = await db.execute_fetchall(
                "SELECT parsed_json FROM ai_decisions WHERE pipeline_run_id = ?",
                (run_id,)
            )
            rationale = "Unknown"
            if ai_rows and ai_rows[0][0]:
                try:
                    parsed = json.loads(ai_rows[0][0])
                    rationale = parsed.get("rationale", "Unknown")
                except json.JSONDecodeError:
                    pass

            # 3. Fetch the market features at entry
            feat_rows = await db.execute_fetchall(
                "SELECT features_json FROM feature_snapshots WHERE pipeline_run_id = ?",
                (run_id,)
            )
            features = feat_rows[0][0] if feat_rows else "{}"

        # 4. Prompt Gemini for a lesson
        prompt = f"""
You are a professional quantitative trader reviewing a past trade.
You must extract ONE concise, generalized trading rule so you do not repeat mistakes or so you double-down on what works.

TRADE OUTCOME:
Symbol: {symbol}
Side: {side}
Entry: ${entry}
Exit: ${exit_px}
PnL: {pnl} USDT
Reason for close: {reason}
Duration: {open_t} to {close_t}

YOUR ORIGINAL RATIONALE FOR ENTERING:
{rationale}

MARKET FEATURES AT ENTRY:
{features}

Analyze the market features versus the outcome.
If it was a loss, what indicator warned you but you ignored it?
If it was a win, what specific combination of indicators made it successful?

Reply ONLY with the extracted trading lesson (max 2 sentences). Start with "LESSON: ".
"""
        
        # Use the raw aiohttp client from AiClient to send the text prompt
        # The AiClient uses the structured output format normally, but here we just want text.
        # We can construct a simple message payload.
        
        lesson_text_raw = await ai_client.request_text(prompt, temperature=0.2)
        if not lesson_text_raw:
            logger.warning("failed to get reflection response")
            return
        lesson_text = lesson_text_raw
        if lesson_text.upper().startswith("LESSON:"):
            lesson_text = lesson_text[7:].strip()

        # 5. Save the lesson
        lesson = TradingLesson(
            id=str(uuid.uuid4()),
            symbol=symbol,
            trade_id=trade_id,
            pnl_usdt=float(pnl or 0.0),
            lesson_text=lesson_text,
            created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        async with get_connection() as db:
            await repo.insert_lesson(db, lesson)
            
        logger.info("extracted new trading lesson for %s: %s", symbol, lesson_text)
            
    except Exception as exc:
        logger.error("trade postmortem failed: %s", exc)
