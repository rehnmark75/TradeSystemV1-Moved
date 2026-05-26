"""
Trade postmortem service — orchestrates data gathering, agent call, and DB persistence.

Call `schedule_postmortem(trade_id, environment)` after a trade's P&L is finalised.
It runs the full pipeline in a fire-and-forget asyncio task.
"""

import json
import logging
import asyncio
from typing import Any, Dict

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trade_postmortem (
    id                  SERIAL PRIMARY KEY,
    trade_id            INTEGER     NOT NULL,
    environment         VARCHAR(10) NOT NULL DEFAULT 'demo',
    generated_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,

    entry_verdict       VARCHAR(20),
    exit_verdict        VARCHAR(20),
    trailing_verdict    VARCHAR(20),

    entry_notes         TEXT,
    exit_notes          TEXT,
    trailing_notes      TEXT,
    key_lesson          TEXT,

    config_delta        JSONB,
    tags                JSONB,

    agent_model         VARCHAR(50),
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_write_tokens  INTEGER,

    UNIQUE (trade_id, environment)
);
"""

_ENSURE_TABLE_DONE = False


def _ensure_table(db: Session) -> None:
    global _ENSURE_TABLE_DONE
    if _ENSURE_TABLE_DONE:
        return
    try:
        db.execute(text(_CREATE_TABLE_SQL))
        db.commit()
        _ENSURE_TABLE_DONE = True
    except Exception as e:
        logger.warning(f"trade_postmortem table ensure failed (may already exist): {e}")
        db.rollback()


def _has_postmortem(db: Session, trade_id: int, environment: str) -> bool:
    row = db.execute(
        text("SELECT 1 FROM trade_postmortem WHERE trade_id = :tid AND environment = :env LIMIT 1"),
        {"tid": trade_id, "env": environment},
    ).fetchone()
    return row is not None


def _save(db: Session, trade_id: int, environment: str, result: Dict[str, Any]) -> None:
    meta = result.pop("_meta", {})
    db.execute(
        text("""
            INSERT INTO trade_postmortem (
                trade_id, environment, generated_at,
                entry_verdict, exit_verdict, trailing_verdict,
                entry_notes, exit_notes, trailing_notes, key_lesson,
                config_delta, tags,
                agent_model, input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens
            ) VALUES (
                :trade_id, :environment, NOW(),
                :entry_verdict, :exit_verdict, :trailing_verdict,
                :entry_notes, :exit_notes, :trailing_notes, :key_lesson,
                :config_delta, :tags,
                :agent_model, :input_tokens, :output_tokens,
                :cache_read_tokens, :cache_write_tokens
            )
            ON CONFLICT (trade_id, environment) DO NOTHING
        """),
        {
            "trade_id": trade_id,
            "environment": environment,
            "entry_verdict": result.get("entry_verdict"),
            "exit_verdict": result.get("exit_verdict"),
            "trailing_verdict": result.get("trailing_verdict"),
            "entry_notes": result.get("entry_notes"),
            "exit_notes": result.get("exit_notes"),
            "trailing_notes": result.get("trailing_notes"),
            "key_lesson": result.get("key_lesson"),
            "config_delta": json.dumps(result.get("config_delta") or {}),
            "tags": json.dumps(result.get("tags") or []),
            "agent_model": meta.get("model"),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
            "cache_read_tokens": meta.get("cache_read_tokens"),
            "cache_write_tokens": meta.get("cache_write_tokens"),
        },
    )
    db.commit()
    logger.info(
        f"✅ Postmortem saved for trade {trade_id} "
        f"(in={meta.get('input_tokens',0)} cached={meta.get('cache_read_tokens',0)} out={meta.get('output_tokens',0)})"
    )


def get_postmortem(db: Session, trade_id: int, environment: str) -> Dict[str, Any] | None:
    """Fetch a stored postmortem row as a plain dict, or None if not found."""
    row = db.execute(
        text("""
            SELECT entry_verdict, exit_verdict, trailing_verdict,
                   entry_notes, exit_notes, trailing_notes, key_lesson,
                   config_delta, tags, generated_at,
                   agent_model, input_tokens, output_tokens
            FROM trade_postmortem
            WHERE trade_id = :tid AND environment = :env
            LIMIT 1
        """),
        {"tid": trade_id, "env": environment},
    ).fetchone()

    if not row:
        return None

    return {
        "entry_verdict": row[0],
        "exit_verdict": row[1],
        "trailing_verdict": row[2],
        "entry_notes": row[3],
        "exit_notes": row[4],
        "trailing_notes": row[5],
        "key_lesson": row[6],
        "config_delta": row[7],
        "tags": row[8],
        "generated_at": str(row[9]) if row[9] else None,
        "agent_model": row[10],
        "input_tokens": row[11],
        "output_tokens": row[12],
    }


async def _run_pipeline(trade_id: int, environment: str) -> None:
    """Full async pipeline: load data → call agent → save."""
    from services.db import SessionLocal
    from services.models import TradeLog, AlertHistory, IGCandle
    from services.trade_analysis_service import (
        calculate_mfe_mae,
        classify_exit_type,
        assess_entry_quality,
        assess_exit_quality,
    )
    from services.trade_postmortem_agent import generate_postmortem

    db: Session = SessionLocal()
    try:
        _ensure_table(db)

        if _has_postmortem(db, trade_id, environment):
            logger.debug(f"Postmortem already exists for trade {trade_id}, skipping")
            return

        trade = db.query(TradeLog).filter(
            TradeLog.id == trade_id,
            TradeLog.environment == environment,
        ).first()

        if not trade or trade.status != "closed":
            logger.debug(f"Trade {trade_id} not closed (status={getattr(trade,'status',None)}), skipping postmortem")
            return

        # Load alert
        alert = None
        strategy_indicators: Dict[str, Any] = {}
        if trade.alert_id:
            alert = db.query(AlertHistory).filter(AlertHistory.id == trade.alert_id).first()
            if alert and alert.strategy_indicators:
                raw = alert.strategy_indicators
                strategy_indicators = json.loads(raw) if isinstance(raw, str) else (raw or {})

        # MFE/MAE
        mfe_mae = calculate_mfe_mae(trade, db, IGCandle, timeframe=1)

        # Quality assessments
        exit_type = classify_exit_type(trade)
        entry_quality = assess_entry_quality(alert, strategy_indicators)
        exit_quality = assess_exit_quality(trade, mfe_mae, exit_type)

        # Duration
        duration_min = 0
        if trade.timestamp and trade.closed_at:
            duration_min = int((trade.closed_at - trade.timestamp).total_seconds() / 60)
        elif trade.lifecycle_duration_minutes:
            duration_min = int(trade.lifecycle_duration_minutes)

        trade_data = {
            "symbol": trade.symbol,
            "direction": trade.direction,
            "profit_loss": float(trade.profit_loss) if trade.profit_loss is not None else 0.0,
            "moved_to_breakeven": bool(trade.moved_to_breakeven),
            "moved_to_stage1": bool(trade.moved_to_stage1),
            "moved_to_stage2": bool(trade.moved_to_stage2),
            "duration_minutes": duration_min,
        }

        alert_data: Dict[str, Any] = {}
        if alert:
            alert_data = {
                "confidence_score": float(alert.confidence_score) if alert.confidence_score else 0.0,
                "market_session": alert.market_session,
                "market_regime": alert.market_regime,
                "strategy": alert.strategy,
            }

        trailing_config: Dict[str, Any] = {}
        try:
            from config import get_trailing_config_for_epic
            trailing_config = get_trailing_config_for_epic(trade.symbol, is_scalp_trade=True) or {}
        except Exception:
            pass

        result = await generate_postmortem(
            trade_data, alert_data, mfe_mae, entry_quality, exit_quality, trailing_config
        )

        if result:
            _save(db, trade_id, environment, result)
        else:
            logger.warning(f"Postmortem agent returned None for trade {trade_id}")

    except Exception as e:
        import traceback
        logger.error(f"Postmortem pipeline error for trade {trade_id}: {e}\n{traceback.format_exc()}")
    finally:
        db.close()


def schedule_postmortem(trade_id: int, environment: str = "demo") -> None:
    """
    Fire-and-forget: schedule the postmortem pipeline as an asyncio task.
    Safe to call from any async context after a trade's P&L is committed.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_run_pipeline(trade_id, environment))
        else:
            logger.warning(f"No running event loop — postmortem for trade {trade_id} skipped")
    except Exception as e:
        logger.error(f"Failed to schedule postmortem for trade {trade_id}: {e}")
