"""Adaptive bucket gate for SMC_SIMPLE live risk control.

The gate learns from recent closed trades in coarse buckets such as
``epic + direction``. Buckets are never permanently disabled: poor recent
performance pauses them, then timed probes allow them to reopen when outcomes
improve.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


@dataclass
class BucketState:
    paused_until: Optional[datetime] = None
    probe_after: Optional[datetime] = None
    last_reason: str = ""


_states: Dict[str, BucketState] = {}
_cache: Dict[str, Tuple[datetime, List[Dict[str, Any]]]] = {}


def _cfg(config: Any, key: str, default: Any, epic: str = "") -> Any:
    pair_overrides = getattr(config, "_pair_overrides", {}) if config else {}
    params = pair_overrides.get(epic, {}).get("parameter_overrides", {}) if epic else {}
    if key in params:
        return params[key]
    return getattr(config, key, default) if config else default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "active", "block"}
    return bool(value)


def _dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        os.getenv("TRADING_DSN", "host=postgres dbname=forex user=postgres password=postgres"),
    )


def _session_from_signal(signal: Dict[str, Any], now: datetime) -> str:
    session = signal.get("market_session") or signal.get("session")
    if session:
        return str(session).lower()
    hour = now.hour
    if 7 <= hour <= 11:
        return "european"
    if 12 <= hour <= 21:
        return "american"
    return "asian"


def _bucket_key(signal: Dict[str, Any], bucket_mode: str, now: datetime) -> str:
    epic = str(signal.get("epic", ""))
    direction = str(signal.get("signal_type") or signal.get("direction") or "").upper()
    session = _session_from_signal(signal, now)
    regime = str(
        signal.get("market_regime_detected")
        or signal.get("market_regime")
        or signal.get("regime")
        or "unknown"
    ).lower()

    if bucket_mode == "direction_session":
        return f"{epic}:{direction}:{session}"
    if bucket_mode == "direction_regime":
        return f"{epic}:{direction}:{regime}"
    return f"{epic}:{direction}"


def _deterministic_probe(signal: Dict[str, Any], bucket_key: str, rate: float) -> bool:
    if rate <= 0:
        return False
    raw = f"{bucket_key}:{signal.get('price')}:{signal.get('timestamp')}:{datetime.utcnow().date()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < rate


def _load_closed_bucket_trades(
    signal: Dict[str, Any],
    config: Any,
    bucket_mode: str,
    window: int,
    environment: str,
) -> List[Dict[str, Any]]:
    epic = str(signal.get("epic", ""))
    direction = str(signal.get("signal_type") or signal.get("direction") or "").upper()
    cache_ttl = int(_cfg(config, "adaptive_bucket_gate_cache_ttl_seconds", 300, epic))
    live_since = _cfg(config, "adaptive_bucket_gate_live_since", None, epic)
    now = datetime.utcnow()
    cache_key = f"{environment}:{bucket_mode}:{epic}:{direction}:{window}:{live_since or ''}"
    cached = _cache.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < cache_ttl:
        return cached[1]

    where_extra = ""
    params: List[Any] = [epic, "SMC_SIMPLE", direction, environment, environment]
    if live_since not in (None, ""):
        where_extra += " AND t.closed_at >= %s "
        params.append(live_since)

    if bucket_mode == "direction_session":
        session = _session_from_signal(signal, now)
        where_extra += """
          AND lower(coalesce(
              nullif(a.market_session, ''),
              CASE
                  WHEN extract(hour from t.closed_at) BETWEEN 7 AND 11 THEN 'european'
                  WHEN extract(hour from t.closed_at) BETWEEN 12 AND 21 THEN 'american'
                  ELSE 'asian'
              END
          )) = %s
        """
        params.append(session)
    elif bucket_mode == "direction_regime":
        regime = str(signal.get("market_regime_detected") or signal.get("market_regime") or "").lower()
        where_extra += "AND lower(coalesce(a.market_regime_detected, a.market_regime, '')) = %s"
        params.append(regime)

    params.append(window)
    query = f"""
        SELECT
            t.closed_at,
            t.pips_gained,
            t.profit_loss
        FROM trade_log t
        JOIN alert_history a ON a.id = t.alert_id
        WHERE t.symbol = %s
          AND a.strategy = %s
          AND a.signal_type = %s
          AND t.status = 'closed'
          AND t.closed_at IS NOT NULL
          AND coalesce(t.pips_gained, t.profit_loss) IS NOT NULL
          AND coalesce(t.environment, a.environment, %s) = %s
          {where_extra}
        ORDER BY t.closed_at DESC
        LIMIT %s
    """

    try:
        conn = psycopg2.connect(_dsn())
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("[AdaptiveBucketGate] trade query failed: %s", exc)
        rows = []

    _cache[cache_key] = (now, rows)
    return rows


def _load_seed_bucket_trades(
    signal: Dict[str, Any],
    config: Any,
    bucket_mode: str,
    window: int,
) -> List[Dict[str, Any]]:
    """Load historical backtest outcomes for bootstrapping live bucket state."""
    epic = str(signal.get("epic", ""))
    direction = str(signal.get("signal_type") or signal.get("direction") or "").upper()
    execution_id = _cfg(config, "adaptive_bucket_gate_seed_execution_id", None, epic)
    if execution_id in (None, "", 0, "0"):
        return []

    strategy_name = str(_cfg(config, "adaptive_bucket_gate_seed_strategy_name", "SMC_SIMPLE", epic))
    cache_ttl = int(_cfg(config, "adaptive_bucket_gate_cache_ttl_seconds", 300, epic))
    now = datetime.utcnow()
    cache_key = f"seed:{execution_id}:{strategy_name}:{bucket_mode}:{epic}:{direction}:{window}"
    cached = _cache.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < cache_ttl:
        return cached[1]

    where_extra = ""
    params: List[Any] = [int(execution_id), epic, strategy_name, direction]

    if bucket_mode == "direction_session":
        session = _session_from_signal(signal, now)
        where_extra = """
          AND (
              CASE
                  WHEN extract(hour from signal_timestamp) BETWEEN 7 AND 11 THEN 'european'
                  WHEN extract(hour from signal_timestamp) BETWEEN 12 AND 21 THEN 'american'
                  ELSE 'asian'
              END
          ) = %s
        """
        params.append(session)
    elif bucket_mode == "direction_regime":
        regime = str(signal.get("market_regime_detected") or signal.get("market_regime") or "").lower()
        where_extra = """
          AND lower(coalesce(
              market_intelligence->>'market_regime_detected',
              market_intelligence->>'market_regime',
              indicator_values->>'market_regime_detected',
              indicator_values->>'market_regime',
              ''
          )) = %s
        """
        params.append(regime)

    params.append(window)
    query = f"""
        SELECT
            signal_timestamp AS closed_at,
            pips_gained,
            NULL::numeric AS profit_loss
        FROM backtest_signals
        WHERE execution_id = %s
          AND epic = %s
          AND strategy_name = %s
          AND signal_type = %s
          AND pips_gained IS NOT NULL
          {where_extra}
        ORDER BY signal_timestamp DESC
        LIMIT %s
    """

    try:
        conn = psycopg2.connect(_dsn())
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("[AdaptiveBucketGate] seed backtest query failed: %s", exc)
        rows = []

    _cache[cache_key] = (now, rows)
    return rows


def _evaluate_rows(rows: List[Dict[str, Any]]) -> Tuple[int, float, Optional[float], int]:
    wins = 0
    pip_values: List[float] = []
    for row in rows:
        pips = row.get("pips_gained")
        pnl = row.get("profit_loss")
        value = pips if pips is not None else pnl
        if value is None:
            continue
        numeric = float(value)
        if numeric > 0:
            wins += 1
        if pips is not None:
            pip_values.append(float(pips))

    n = len(rows)
    win_rate = wins / n if n else 0.0
    expectancy = sum(pip_values) / len(pip_values) if pip_values else None
    return n, win_rate, expectancy, len(pip_values)


def apply_adaptive_bucket_gate(
    signal: Optional[Dict[str, Any]],
    config: Any,
    strategy_logger: Optional[logging.Logger] = None,
    *,
    backtest_mode: bool = False,
    environment: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the signal if allowed, or ``None`` when blocked in ACTIVE mode."""
    if signal is None:
        return None

    log = strategy_logger or logger
    epic = str(signal.get("epic", ""))
    enabled = _bool(_cfg(config, "adaptive_bucket_gate_enabled", False, epic))
    if not enabled:
        signal.setdefault("adaptive_bucket_gate_state", "disabled")
        return signal

    if backtest_mode:
        signal.setdefault("adaptive_bucket_gate_state", "backtest_disabled")
        return signal

    mode = str(_cfg(config, "adaptive_bucket_gate_mode", "MONITORING", epic)).upper()
    bucket_mode = str(_cfg(config, "adaptive_bucket_gate_bucket_mode", "direction", epic))
    window = int(_cfg(config, "adaptive_bucket_gate_window", 12, epic))
    min_trades = int(_cfg(config, "adaptive_bucket_gate_min_trades", 8, epic))
    min_wr = float(_cfg(config, "adaptive_bucket_gate_min_win_rate", 0.45, epic))
    min_expectancy = float(_cfg(config, "adaptive_bucket_gate_min_expectancy_pips", -0.25, epic))
    min_pip_samples = int(_cfg(config, "adaptive_bucket_gate_min_pip_samples", 4, epic))
    pause_hours = float(_cfg(config, "adaptive_bucket_gate_pause_hours", 48, epic))
    probe_after_hours = float(_cfg(config, "adaptive_bucket_gate_probe_after_hours", 12, epic))
    exploration_rate = float(_cfg(config, "adaptive_bucket_gate_exploration_rate", 0.02, epic))
    max_confidence_raw = _cfg(config, "adaptive_bucket_gate_max_confidence", None, epic)
    max_confidence = float(max_confidence_raw) if max_confidence_raw not in (None, "") else None
    env = environment or os.getenv("TRADING_ENVIRONMENT", "demo")

    now = datetime.utcnow()
    key = _bucket_key(signal, bucket_mode, now)
    state = _states.setdefault(key, BucketState())

    live_rows = _load_closed_bucket_trades(signal, config, bucket_mode, window, env)
    rows = live_rows
    stats_source = "live"
    if len(live_rows) < min_trades:
        seed_window = int(_cfg(config, "adaptive_bucket_gate_seed_window", window, epic))
        seed_rows = _load_seed_bucket_trades(signal, config, bucket_mode, seed_window)
        if seed_rows:
            rows = (live_rows + seed_rows)[:max(window, len(live_rows))]
            stats_source = "live_seed_backtest" if live_rows else "seed_backtest"

    n, win_rate, expectancy, pip_samples = _evaluate_rows(rows)

    poor_win_rate = n >= min_trades and win_rate < min_wr
    poor_expectancy = pip_samples >= min_pip_samples and expectancy is not None and expectancy < min_expectancy
    weak_bucket = poor_win_rate and (poor_expectancy or pip_samples < min_pip_samples)

    if not weak_bucket and n >= min_trades:
        state.paused_until = None
        state.probe_after = None
        state.last_reason = ""

    if weak_bucket and state.paused_until is None:
        state.paused_until = now + timedelta(hours=pause_hours)
        state.probe_after = now + timedelta(hours=probe_after_hours)
        exp_text = "n/a" if expectancy is None else f"{expectancy:.2f}"
        state.last_reason = (
            f"{key}: WR={win_rate:.0%} on {n} closed trades, "
            f"expectancy={exp_text} pips"
        )

    paused = state.paused_until is not None and now < state.paused_until
    probing = paused and state.probe_after is not None and now >= state.probe_after
    probe_allowed = probing and _deterministic_probe(signal, key, exploration_rate)

    signal["adaptive_bucket_gate_state"] = "paused" if paused else "open"
    signal["adaptive_bucket_gate_bucket"] = key
    signal["adaptive_bucket_gate_win_rate"] = win_rate
    signal["adaptive_bucket_gate_trades"] = n
    signal["adaptive_bucket_gate_expectancy_pips"] = expectancy
    signal["adaptive_bucket_gate_stats_source"] = stats_source

    confidence = signal.get("confidence_score", signal.get("confidence"))
    if max_confidence is not None and confidence is not None and float(confidence) > max_confidence:
        reason = f"{key}: confidence {float(confidence):.0%} > cap {max_confidence:.0%}"
        signal["adaptive_bucket_gate_state"] = "confidence_capped"
        signal["adaptive_bucket_gate_would_block"] = True
        signal["adaptive_bucket_gate_reason"] = reason
        log.warning("[AdaptiveBucketGate] %s (%s mode)", reason, mode)
        if mode == "ACTIVE":
            return None
        return signal

    if paused and not probe_allowed:
        reason = state.last_reason or f"{key}: paused by adaptive bucket gate"
        signal["adaptive_bucket_gate_would_block"] = True
        signal["adaptive_bucket_gate_reason"] = reason
        log.warning("[AdaptiveBucketGate] %s (%s mode)", reason, mode)
        if mode == "ACTIVE":
            return None
        return signal

    if probe_allowed:
        signal["adaptive_bucket_gate_state"] = "probe"
        signal["adaptive_bucket_gate_probe"] = True
        log.info("[AdaptiveBucketGate] Probe allowed for %s after pause", key)

    return signal
