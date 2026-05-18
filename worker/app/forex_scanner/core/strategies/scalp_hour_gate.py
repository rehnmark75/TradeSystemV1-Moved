"""Dynamic hour gate for SMC_SIMPLE scalp entries.

Static blocked hours are useful as a baseline, but they should not become
permanent. This gate overlays recent realized performance by epic + UTC hour
and can release recovered static hours or block newly weak hours.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

_cache: Dict[str, tuple[datetime, List[Dict[str, Any]]]] = {}


@dataclass
class HourGateDecision:
    allowed: bool
    state: str
    reason: str
    static_blocked: bool
    dynamic_blocked: bool
    dynamic_released: bool
    probe: bool
    sample_size: int
    win_rate: float
    profit_factor: Optional[float]
    expectancy_pips: Optional[float]
    stats_source: str

    def as_context(self) -> Dict[str, Any]:
        return {
            "filter": "dynamic_scalp_hour_gate",
            "state": self.state,
            "reason": self.reason,
            "static_blocked": self.static_blocked,
            "dynamic_blocked": self.dynamic_blocked,
            "dynamic_released": self.dynamic_released,
            "probe": self.probe,
            "sample_size": self.sample_size,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy_pips": self.expectancy_pips,
            "stats_source": self.stats_source,
        }


def _cfg(config: Any, key: str, default: Any, epic: str = "") -> Any:
    pair_overrides = getattr(config, "_pair_overrides", {}) if config else {}
    pair_data = pair_overrides.get(epic, {}) if epic else {}
    if key in pair_data and pair_data[key] is not None:
        return pair_data[key]
    params = pair_data.get("parameter_overrides", {}) if pair_data else {}
    if key in params:
        return params[key]
    return getattr(config, key, default) if config else default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "active"}
    return bool(value)


def _dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        os.getenv("TRADING_DSN", "host=postgres dbname=forex user=postgres password=postgres"),
    )


def _deterministic_probe(epic: str, hour_utc: int, timestamp: Optional[datetime], rate: float) -> bool:
    if rate <= 0:
        return False
    day = (timestamp or datetime.utcnow()).date()
    raw = f"{epic}:{hour_utc}:{day}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < rate


def _load_live_rows(
    epic: str,
    hour_utc: int,
    lookback_days: int,
    window: int,
    environment: str,
    cache_ttl_seconds: int,
) -> List[Dict[str, Any]]:
    now = datetime.utcnow()
    cache_key = f"live:{environment}:{epic}:{hour_utc}:{lookback_days}:{window}"
    cached = _cache.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < cache_ttl_seconds:
        return cached[1]

    query = """
        SELECT
            t.closed_at,
            t.pips_gained,
            t.profit_loss
        FROM trade_log t
        JOIN alert_history a ON a.id = t.alert_id
        WHERE t.symbol = %s
          AND a.strategy = 'SMC_SIMPLE'
          AND t.status = 'closed'
          AND t.closed_at IS NOT NULL
          AND t.closed_at >= NOW() - (%s::text || ' days')::interval
          AND extract(hour from t.closed_at) = %s
          AND coalesce(t.pips_gained, t.profit_loss) IS NOT NULL
          AND coalesce(t.environment, a.environment, %s) = %s
        ORDER BY t.closed_at DESC
        LIMIT %s
    """
    rows: List[Dict[str, Any]] = []
    try:
        conn = psycopg2.connect(_dsn())
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, [epic, lookback_days, hour_utc, environment, environment, window])
                rows = [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("[ScalpHourGate] live trade query failed: %s", exc)

    _cache[cache_key] = (now, rows)
    return rows


def _load_seed_rows(
    epic: str,
    hour_utc: int,
    config: Any,
    signal_timestamp: Optional[datetime],
    window: int,
    cache_ttl_seconds: int,
) -> List[Dict[str, Any]]:
    execution_id = _cfg(config, "scalp_hour_gate_seed_execution_id", None, epic)
    if execution_id in (None, "", 0, "0"):
        return []

    strategy_name = str(_cfg(config, "scalp_hour_gate_seed_strategy_name", "SMC_SIMPLE", epic))
    before_ts = signal_timestamp or datetime.utcnow()
    cache_key = f"seed:{execution_id}:{strategy_name}:{epic}:{hour_utc}:{before_ts.isoformat()}:{window}"
    cached = _cache.get(cache_key)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < cache_ttl_seconds:
        return cached[1]

    query = """
        SELECT
            signal_timestamp AS closed_at,
            pips_gained,
            NULL::numeric AS profit_loss
        FROM backtest_signals
        WHERE execution_id = %s
          AND epic = %s
          AND strategy_name = %s
          AND extract(hour from signal_timestamp) = %s
          AND signal_timestamp < %s
          AND pips_gained IS NOT NULL
        ORDER BY signal_timestamp DESC
        LIMIT %s
    """
    rows: List[Dict[str, Any]] = []
    try:
        conn = psycopg2.connect(_dsn())
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, [int(execution_id), epic, strategy_name, hour_utc, before_ts, window])
                rows = [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("[ScalpHourGate] seed backtest query failed: %s", exc)

    _cache[cache_key] = (datetime.utcnow(), rows)
    return rows


def _stats(rows: List[Dict[str, Any]]) -> tuple[int, float, Optional[float], Optional[float]]:
    values: List[float] = []
    for row in rows:
        value = row.get("pips_gained")
        if value is None:
            value = row.get("profit_loss")
        if value is not None:
            values.append(float(value))

    n = len(values)
    if n == 0:
        return 0, 0.0, None, None

    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0
    return n, len(wins) / n, profit_factor, sum(values) / n


def evaluate_scalp_hour_gate(
    *,
    epic: str,
    pair: str,
    hour_utc: int,
    static_blocked_hours: Optional[List[int]],
    config: Any,
    signal_timestamp: Optional[datetime],
    backtest_mode: bool,
    environment: Optional[str] = None,
) -> HourGateDecision:
    static_blocked = hour_utc in set(static_blocked_hours or [])
    enabled = _bool(_cfg(config, "scalp_hour_gate_enabled", True, epic))
    mode = str(_cfg(config, "scalp_hour_gate_mode", "ACTIVE", epic)).upper()
    if not enabled:
        return HourGateDecision(
            allowed=not static_blocked,
            state="static_only",
            reason=f"static hour block {'matched' if static_blocked else 'not matched'}",
            static_blocked=static_blocked,
            dynamic_blocked=False,
            dynamic_released=False,
            probe=False,
            sample_size=0,
            win_rate=0.0,
            profit_factor=None,
            expectancy_pips=None,
            stats_source="disabled",
        )

    window = int(_cfg(config, "scalp_hour_gate_window", 24, epic))
    min_trades = int(_cfg(config, "scalp_hour_gate_min_trades", 8, epic))
    lookback_days = int(_cfg(config, "scalp_hour_gate_lookback_days", 45, epic))
    block_pf = float(_cfg(config, "scalp_hour_gate_block_profit_factor", 0.75, epic))
    block_expectancy = float(_cfg(config, "scalp_hour_gate_block_expectancy_pips", -0.15, epic))
    release_pf = float(_cfg(config, "scalp_hour_gate_release_profit_factor", 1.05, epic))
    release_expectancy = float(_cfg(config, "scalp_hour_gate_release_expectancy_pips", 0.0, epic))
    probe_rate = float(_cfg(config, "scalp_hour_gate_probe_rate", 0.10, epic))
    cache_ttl_seconds = int(_cfg(config, "scalp_hour_gate_cache_ttl_seconds", 300, epic))
    env = environment or os.getenv("TRADING_ENVIRONMENT", "demo")

    rows: List[Dict[str, Any]]
    stats_source: str
    if backtest_mode:
        rows = _load_seed_rows(epic, hour_utc, config, signal_timestamp, window, cache_ttl_seconds)
        stats_source = "seed_backtest" if rows else "backtest_no_seed"
    else:
        rows = _load_live_rows(epic, hour_utc, lookback_days, window, env, cache_ttl_seconds)
        stats_source = "live"
        if len(rows) < min_trades:
            seed_rows = _load_seed_rows(epic, hour_utc, config, signal_timestamp, window, cache_ttl_seconds)
            if seed_rows:
                rows = (rows + seed_rows)[:window]
                stats_source = "live_seed_backtest"

    sample_size, win_rate, profit_factor, expectancy = _stats(rows)
    has_evidence = sample_size >= min_trades and profit_factor is not None and expectancy is not None
    dynamic_blocked = has_evidence and profit_factor < block_pf and expectancy < block_expectancy
    dynamic_released = has_evidence and profit_factor >= release_pf and expectancy >= release_expectancy
    probe = False

    if dynamic_blocked:
        allowed = mode != "ACTIVE"
        state = "dynamic_block"
    elif backtest_mode and stats_source == "backtest_no_seed" and static_blocked:
        # Without a seed execution, enforcing the static block in historical
        # replay prevents us from discovering whether the hour has recovered.
        # Live still enforces the static block/probe path using live outcomes.
        allowed = True
        state = "backtest_no_seed_open"
    elif static_blocked and dynamic_released:
        allowed = True
        state = "dynamic_release"
    elif static_blocked:
        probe = _deterministic_probe(epic, hour_utc, signal_timestamp, probe_rate)
        allowed = probe or mode != "ACTIVE"
        state = "static_probe" if probe else "static_block"
    else:
        allowed = True
        state = "open"

    pf_text = "n/a" if profit_factor is None else f"{profit_factor:.2f}"
    exp_text = "n/a" if expectancy is None else f"{expectancy:.2f}"
    reason = (
        f"Scalp hour gate {state} for {pair} hour={hour_utc} UTC "
        f"(static={static_blocked}, n={sample_size}, wr={win_rate:.0%}, "
        f"pf={pf_text}, expectancy={exp_text}, source={stats_source})"
    )
    return HourGateDecision(
        allowed=allowed,
        state=state,
        reason=reason,
        static_blocked=static_blocked,
        dynamic_blocked=dynamic_blocked,
        dynamic_released=dynamic_released,
        probe=probe,
        sample_size=sample_size,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy_pips=expectancy,
        stats_source=stats_source,
    )
