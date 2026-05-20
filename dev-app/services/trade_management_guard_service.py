"""
Trade Management Guard Service

Evaluates post-entry exit rules (guards) for open trades each monitoring cycle.
Guards are configured in strategy_config.trade_management_guards and are fully
DB-driven — mode toggle (monitor → active) requires only an UPDATE, no deploy.

Currently implemented guard type: 'failed_followthrough'
  Close a BUY trade early when:
    - trade age <= max_age_minutes
    - MFE (peak profit) < min_mfe_pips
    - current adverse excursion >= adverse_trigger_pips

Adding a new guard type:
  1. Insert a row in trade_management_guards with a new guard_type value.
  2. Add an evaluation branch in _evaluate_guard() below.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import httpx
import psycopg2
import psycopg2.extras

logger = logging.getLogger("trade_monitor")

# ---- strategy_config DB connection -----------------------------------------
import os

_STRATEGY_CONFIG_DSN = os.getenv(
    "STRATEGY_CONFIG_DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/strategy_config"
)

# ---- Cache -----------------------------------------------------------------
_GUARD_CACHE: List[Dict[str, Any]] = []
_CACHE_LOADED_AT: Optional[datetime] = None
_CACHE_TTL_SECONDS = 300  # 5-minute TTL


def _load_guards() -> List[Dict[str, Any]]:
    """Load enabled guards from strategy_config DB (cached for 5 min)."""
    global _GUARD_CACHE, _CACHE_LOADED_AT

    now = datetime.now(timezone.utc)
    if _CACHE_LOADED_AT and (now - _CACHE_LOADED_AT).total_seconds() < _CACHE_TTL_SECONDS:
        return _GUARD_CACHE

    try:
        conn = psycopg2.connect(_STRATEGY_CONFIG_DSN)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, name, guard_type, mode,
                       applies_to_strategies, epic_filter, direction_filter,
                       condition_config
                FROM trade_management_guards
                WHERE enabled = TRUE
            """)
            rows = cur.fetchall()
        conn.close()
        _GUARD_CACHE = [dict(r) for r in rows]
        _CACHE_LOADED_AT = now
        logger.debug(f"[GUARD] Loaded {len(_GUARD_CACHE)} enabled guard(s) from DB")
    except Exception as exc:
        logger.error(f"[GUARD] Failed to load guards from strategy_config: {exc}")
        # Return stale cache rather than crashing the monitor loop
    return _GUARD_CACHE


def _log_decision(guard: Dict, trade_id: int, deal_id: str, epic: str,
                  direction: str, strategy: str, age_minutes: float,
                  mfe_pips: float, mae_pips: float, executed: bool,
                  close_error: Optional[str] = None) -> None:
    """Append a row to trade_management_decisions."""
    try:
        conn = psycopg2.connect(_STRATEGY_CONFIG_DSN)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trade_management_decisions
                    (guard_id, trade_id, deal_id, epic, direction, strategy,
                     mode_at_firing, age_minutes, mfe_pips, mae_pips,
                     executed, close_error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                guard["id"], trade_id, deal_id, epic, direction, strategy,
                guard["mode"], round(age_minutes, 2), round(mfe_pips, 2),
                round(mae_pips, 2), executed, close_error,
            ))
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.error(f"[GUARD] Failed to log decision for trade {trade_id}: {exc}")


# ---- Guard evaluation ------------------------------------------------------

def _get_strategy(trade, db) -> str:
    """Resolve strategy name from trade, falling back to alert_history lookup."""
    direct = getattr(trade, "strategy", None)
    if direct:
        return str(direct).upper()
    if db is not None and getattr(trade, "alert_id", None):
        try:
            from services.models import AlertHistory
            alert = db.query(AlertHistory).filter(
                AlertHistory.id == trade.alert_id
            ).first()
            if alert and getattr(alert, "strategy", None):
                return str(alert.strategy).upper()
        except Exception:
            pass
    symbol = (getattr(trade, "symbol", "") or "").upper()
    if "CFEGOLD" in symbol or "XAU" in symbol or "GOLD" in symbol:
        return "XAU_GOLD"
    return "UNKNOWN"


def _evaluate_failed_followthrough(trade, current_price: float,
                                   cfg: Dict) -> Optional[Dict]:
    """
    Evaluate the 'failed_followthrough' guard for a single trade.

    Returns a dict with evaluation details if the guard fires, else None.
    Uses mid-price (already computed by the monitor loop) for excursion
    math. The 0.3-0.5 pip spread delta is negligible against a 6-pip trigger.
    """
    direction = (trade.direction or "").upper()

    from utils import get_point_value
    point_value = get_point_value(trade.symbol)

    entry = float(trade.entry_price)
    if direction == "BUY":
        current_profit_pips = (current_price - entry) / point_value
    else:
        current_profit_pips = (entry - current_price) / point_value

    favorable_pips = max(0.0, current_profit_pips)
    adverse_pips   = max(0.0, -current_profit_pips)

    # Peak MFE tracked by _update_peak_and_mae; use it as running max
    peak_mfe = max(float(trade.vsl_peak_profit_pips or 0.0), favorable_pips)

    opened_at = trade.timestamp
    if opened_at is not None and opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)
    age_minutes = (datetime.now(timezone.utc) - opened_at).total_seconds() / 60 if opened_at else 9999

    max_age      = float(cfg.get("max_age_minutes", 30))
    min_mfe      = float(cfg.get("min_mfe_pips", 3.0))
    adverse_trig = float(cfg.get("adverse_trigger_pips", 6.0))

    if age_minutes > max_age:
        return None
    if peak_mfe >= min_mfe:
        return None
    if adverse_pips < adverse_trig:
        return None

    return {
        "age_minutes": age_minutes,
        "mfe_pips": peak_mfe,
        "mae_pips": adverse_pips,
    }


def _evaluate_guard(guard: Dict, trade, current_price: float,
                    strategy: str) -> Optional[Dict]:
    """Dispatch to the correct evaluator for guard_type."""
    guard_type = guard.get("guard_type")
    cfg = guard.get("condition_config") or {}

    if guard_type == "failed_followthrough":
        return _evaluate_failed_followthrough(trade, current_price, cfg)

    logger.warning(f"[GUARD] Unknown guard_type '{guard_type}' — skipping")
    return None


def _matches_guard_filters(guard: Dict, trade, strategy: str) -> bool:
    """Check epic, direction, and strategy filters (NULL = match all)."""
    epic_filter = guard.get("epic_filter")
    if epic_filter and trade.symbol not in epic_filter:
        return False

    direction_filter = guard.get("direction_filter")
    if direction_filter and (trade.direction or "").upper() not in direction_filter:
        return False

    strategies = guard.get("applies_to_strategies")
    if strategies and strategy not in strategies:
        return False

    return True


# ---- Position close helper -------------------------------------------------

async def _close_position(trade, trading_headers: dict) -> bool:
    """
    Send a MARKET close order to IG for the given trade.
    Fetches the position from IG to get the exact size before closing.
    Returns True on success.
    """
    from config import API_BASE_URL
    close_direction = "SELL" if (trade.direction or "").upper() == "BUY" else "BUY"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch live position to get current size
            pos_resp = await client.get(
                f"{API_BASE_URL}/positions",
                headers={
                    "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                    "CST": trading_headers["CST"],
                    "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                    "Accept": "application/json",
                    "Version": "2",
                },
            )
            pos_resp.raise_for_status()
            positions = pos_resp.json().get("positions", [])
            ig_pos = next(
                (p for p in positions if p["position"]["dealId"] == trade.deal_id),
                None,
            )
            if not ig_pos:
                logger.error(f"[GUARD CLOSE] Trade {trade.id}: deal {trade.deal_id} not found on IG")
                return False

            size = ig_pos["position"]["size"]

            # Send market close
            close_resp = await client.post(
                f"{API_BASE_URL}/positions/otc",
                headers={
                    "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                    "CST": trading_headers["CST"],
                    "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                    "Content-Type": "application/json",
                    "_method": "DELETE",
                    "Version": "1",
                },
                json={
                    "dealId": trade.deal_id,
                    "direction": close_direction,
                    "orderType": "MARKET",
                    "size": size,
                },
            )
            if close_resp.status_code in (200, 201):
                logger.info(
                    f"[GUARD CLOSE] Trade {trade.id} {trade.symbol}: "
                    f"market close sent (deal={trade.deal_id}, size={size})"
                )
                return True
            else:
                logger.error(
                    f"[GUARD CLOSE] Trade {trade.id}: IG returned {close_resp.status_code} — {close_resp.text}"
                )
                return False

    except Exception as exc:
        logger.error(f"[GUARD CLOSE] Trade {trade.id}: exception during close: {exc}")
        return False


# ---- Public entry point ----------------------------------------------------

async def evaluate_guards_for_trade(
    trade,
    current_price: float,
    db,
    trading_headers: Optional[dict],
) -> bool:
    """
    Evaluate all enabled guards for a single trade.

    Called once per monitoring cycle, after MFE/MAE update and before the
    trailing processor. Returns True if a guard fired an active close
    (caller should skip further processing for this trade).

    Args:
        trade:           TradeLog ORM object (loaded in current session)
        current_price:   Mid-price from the monitor loop
        db:              Active SQLAlchemy session (for strategy lookup)
        trading_headers: IG auth headers (needed for active closes)
    """
    if getattr(trade, "failed_followthrough_exit", False):
        return False  # Already closed by this guard — don't re-fire

    guards = _load_guards()
    if not guards:
        return False

    strategy = _get_strategy(trade, db)

    for guard in guards:
        if not _matches_guard_filters(guard, trade, strategy):
            continue

        result = _evaluate_guard(guard, trade, current_price, strategy)
        if result is None:
            continue

        mode = guard["mode"]
        age     = result["age_minutes"]
        mfe     = result["mfe_pips"]
        mae     = result["mae_pips"]
        executed = False
        close_error = None

        logger.info(
            f"[GUARD:{guard['name']}] Trade {trade.id} {trade.symbol} {trade.direction} | "
            f"mode={mode} age={age:.1f}m MFE={mfe:.1f}p MAE={mae:.1f}p"
        )

        if mode == "active":
            if not trading_headers:
                close_error = "no_trading_headers"
                logger.error(f"[GUARD] Cannot close trade {trade.id} — no trading headers")
            elif not trade.deal_id:
                close_error = "no_deal_id"
                logger.error(f"[GUARD] Cannot close trade {trade.id} — no deal_id")
            else:
                ok = await _close_position(trade, trading_headers)
                if ok:
                    executed = True
                    trade.failed_followthrough_exit    = True
                    trade.failed_followthrough_exit_at = datetime.utcnow()
                    trade.failed_followthrough_mfe_pips = mfe
                    trade.failed_followthrough_mae_pips = mae
                    trade.status     = "closed"
                    trade.exit_reason = "failed_followthrough_exit"
                    trade.trigger_time = datetime.utcnow()
                    db.commit()
                    logger.info(
                        f"[GUARD ACTIVE] Trade {trade.id} closed: "
                        f"MFE={mfe:.1f}p MAE={mae:.1f}p age={age:.1f}m"
                    )
                else:
                    close_error = "close_api_failed"
        else:
            logger.info(
                f"[GUARD MONITOR] Would close trade {trade.id} — "
                f"enable mode='active' to execute"
            )

        _log_decision(
            guard=guard,
            trade_id=trade.id,
            deal_id=trade.deal_id or "",
            epic=trade.symbol,
            direction=(trade.direction or "").upper(),
            strategy=strategy,
            age_minutes=age,
            mfe_pips=mfe,
            mae_pips=mae,
            executed=executed,
            close_error=close_error,
        )

        if executed:
            return True  # Active close succeeded — skip trailing processor

    return False
