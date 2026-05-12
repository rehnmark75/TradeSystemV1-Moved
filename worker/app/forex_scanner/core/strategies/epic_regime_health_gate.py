"""Epic-level regime health gate for SMC_SIMPLE.

This guardrail answers a narrower question than a generic market-regime label:
"Has this exact epic/strategy template been profitable recently under similar
conditions?"  It can run in MONITORING mode to annotate signals, or ACTIVE mode
to block signals when recent realized edge is unfavorable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthMetrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    pips: float = 0.0
    expectancy: float = 0.0
    profit_factor: Optional[float] = None


@dataclass(frozen=True)
class HealthDecision:
    state: str
    reason: str
    execution_id: Optional[int]
    overall: HealthMetrics
    bucket: Optional[HealthMetrics] = None
    bucket_key: Optional[str] = None


_cache: Dict[str, Tuple[datetime, Optional[HealthDecision]]] = {}


def _dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        os.getenv("TRADING_DSN", "host=postgres dbname=forex user=postgres password=postgres"),
    )


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
        return value.strip().lower() in {"1", "true", "yes", "on", "active"}
    return bool(value)


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _signal_ts(signal: Dict[str, Any], fallback: Optional[datetime]) -> datetime:
    raw = signal.get("timestamp") or signal.get("signal_timestamp") or signal.get("candle_timestamp")
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None)
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    return (fallback or datetime.utcnow()).replace(tzinfo=None)


def _session(hour: int) -> str:
    if 7 <= hour <= 11:
        return "european"
    if 12 <= hour <= 21:
        return "american"
    return "asian"


def _bucket_clause(bucket_mode: str, signal: Dict[str, Any], ts: datetime) -> Tuple[str, List[Any], str]:
    direction = str(signal.get("signal_type") or signal.get("direction") or "").upper()
    hour = int(ts.hour)
    session = _session(hour)

    if bucket_mode == "direction_hour":
        return "AND bs.signal_type = %s AND EXTRACT(HOUR FROM bs.signal_timestamp)::int = %s", [direction, hour], f"{direction}:h{hour:02d}"
    if bucket_mode == "direction_session":
        return (
            """AND bs.signal_type = %s
               AND CASE
                   WHEN EXTRACT(HOUR FROM bs.signal_timestamp)::int BETWEEN 7 AND 11 THEN 'european'
                   WHEN EXTRACT(HOUR FROM bs.signal_timestamp)::int BETWEEN 12 AND 21 THEN 'american'
                   ELSE 'asian'
               END = %s""",
            [direction, session],
            f"{direction}:{session}",
        )
    if bucket_mode == "hour":
        return "AND EXTRACT(HOUR FROM bs.signal_timestamp)::int = %s", [hour], f"h{hour:02d}"
    if bucket_mode == "session":
        return (
            """AND CASE
                   WHEN EXTRACT(HOUR FROM bs.signal_timestamp)::int BETWEEN 7 AND 11 THEN 'european'
                   WHEN EXTRACT(HOUR FROM bs.signal_timestamp)::int BETWEEN 12 AND 21 THEN 'american'
                   ELSE 'asian'
               END = %s""",
            [session],
            session,
        )
    return "AND bs.signal_type = %s", [direction], direction


def _metrics(rows: List[Dict[str, Any]]) -> HealthMetrics:
    if not rows:
        return HealthMetrics()
    pips = [float(row["pips_gained"] or 0.0) for row in rows]
    wins = sum(1 for value in pips if value > 0)
    losses = sum(1 for value in pips if value < 0)
    gross_win = sum(value for value in pips if value > 0)
    gross_loss = abs(sum(value for value in pips if value < 0))
    return HealthMetrics(
        trades=len(rows),
        wins=wins,
        losses=losses,
        win_rate=wins / len(rows),
        pips=sum(pips),
        expectancy=sum(pips) / len(rows),
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else None,
    )


def _latest_execution_id(cur, epic: str, strategy_name: str) -> Optional[int]:
    cur.execute(
        """
        SELECT id
        FROM backtest_executions
        WHERE strategy_name = %s
          AND status = 'completed'
          AND %s = ANY(epics_tested)
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (strategy_name, epic),
    )
    row = cur.fetchone()
    return int(row["id"]) if row else None


def _load_rows(
    cur,
    *,
    execution_id: int,
    epic: str,
    strategy_name: str,
    cutoff_ts: Optional[datetime],
    window_days: int,
    recent_trades: int,
    extra_clause: str = "",
    extra_params: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    params: List[Any] = [execution_id, epic, strategy_name]
    cutoff_sql = ""
    if cutoff_ts:
        cutoff_sql = "AND bs.signal_timestamp < %s"
        params.append(cutoff_ts)
    if window_days > 0:
        cutoff = (cutoff_ts or datetime.utcnow()) - timedelta(days=window_days)
        cutoff_sql += " AND bs.signal_timestamp >= %s"
        params.append(cutoff)
    if extra_clause:
        params.extend(extra_params or [])
    params.append(recent_trades)

    cur.execute(
        f"""
        SELECT bs.signal_timestamp, bs.signal_type, bs.pips_gained, bs.trade_result
        FROM backtest_signals bs
        WHERE bs.execution_id = %s
          AND bs.epic = %s
          AND bs.strategy_name = %s
          AND bs.pips_gained IS NOT NULL
          {cutoff_sql}
          {extra_clause}
        ORDER BY bs.signal_timestamp DESC
        LIMIT %s
        """,
        params,
    )
    return [dict(row) for row in cur.fetchall()]


def _classify(
    metrics: HealthMetrics,
    *,
    min_trades: int,
    favorable_min_pf: float,
    favorable_min_expectancy: float,
    favorable_min_wr: float,
    unfavorable_max_pf: float,
    unfavorable_max_expectancy: float,
    unfavorable_max_wr: float,
) -> str:
    if metrics.trades < min_trades:
        return "neutral"
    pf = metrics.profit_factor if metrics.profit_factor is not None else 99.0
    if (
        pf >= favorable_min_pf
        and metrics.expectancy >= favorable_min_expectancy
        and metrics.win_rate >= favorable_min_wr
    ):
        return "favorable"
    if (
        pf < unfavorable_max_pf
        or metrics.expectancy < unfavorable_max_expectancy
        or metrics.win_rate < unfavorable_max_wr
    ):
        return "unfavorable"
    return "neutral"


def evaluate_epic_regime_health(
    signal: Dict[str, Any],
    config: Any,
    *,
    signal_timestamp: Optional[datetime] = None,
    backtest_mode: bool = False,
) -> Optional[HealthDecision]:
    epic = str(signal.get("epic", ""))
    strategy_name = str(signal.get("strategy") or signal.get("strategy_name") or "SMC_SIMPLE")
    ts = _signal_ts(signal, signal_timestamp)

    explicit_execution = _cfg(config, "epic_regime_health_execution_id", None, epic)
    execution_id = int(explicit_execution) if explicit_execution not in (None, "") else None
    source = "explicit" if execution_id else "latest"
    window_days = int(_cfg(config, "epic_regime_health_window_days", 45, epic))
    recent_trades = int(_cfg(config, "epic_regime_health_recent_trades", 40, epic))
    min_trades = int(_cfg(config, "epic_regime_health_min_trades", 12, epic))
    bucket_min_trades = int(_cfg(config, "epic_regime_health_bucket_min_trades", 4, epic))
    bucket_mode = str(_cfg(config, "epic_regime_health_bucket_mode", "direction_hour", epic))
    ttl = int(_cfg(config, "epic_regime_health_cache_ttl_seconds", 300, epic))
    use_cutoff = _bool(_cfg(config, "epic_regime_health_use_signal_cutoff", backtest_mode, epic))

    cache_key = f"{source}:{execution_id}:{epic}:{strategy_name}:{ts.isoformat()}:{bucket_mode}:{window_days}:{recent_trades}:{use_cutoff}"
    cached = _cache.get(cache_key)
    now = datetime.utcnow()
    if cached and (now - cached[0]).total_seconds() < ttl:
        return cached[1]

    try:
        conn = psycopg2.connect(_dsn())
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if execution_id is None:
                    execution_id = _latest_execution_id(cur, epic, strategy_name)
                if execution_id is None:
                    decision = HealthDecision(
                        state="neutral",
                        reason=f"No completed {strategy_name} backtest found for {epic}",
                        execution_id=None,
                        overall=HealthMetrics(),
                    )
                    _cache[cache_key] = (now, decision)
                    return decision

                cutoff_ts = ts if use_cutoff else None
                overall_rows = _load_rows(
                    cur,
                    execution_id=execution_id,
                    epic=epic,
                    strategy_name=strategy_name,
                    cutoff_ts=cutoff_ts,
                    window_days=window_days,
                    recent_trades=recent_trades,
                )
                bucket_clause, bucket_params, bucket_key = _bucket_clause(bucket_mode, signal, ts)
                bucket_rows = _load_rows(
                    cur,
                    execution_id=execution_id,
                    epic=epic,
                    strategy_name=strategy_name,
                    cutoff_ts=cutoff_ts,
                    window_days=window_days,
                    recent_trades=recent_trades,
                    extra_clause=bucket_clause,
                    extra_params=bucket_params,
                )
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("[EpicRegimeHealth] query failed: %s", exc)
        return None

    favorable_min_pf = float(_cfg(config, "epic_regime_health_favorable_min_pf", 1.4, epic))
    favorable_min_exp = float(_cfg(config, "epic_regime_health_favorable_min_expectancy_pips", 1.0, epic))
    favorable_min_wr = float(_cfg(config, "epic_regime_health_favorable_min_win_rate", 0.52, epic))
    unfavorable_max_pf = float(_cfg(config, "epic_regime_health_unfavorable_max_pf", 1.0, epic))
    unfavorable_max_exp = float(_cfg(config, "epic_regime_health_unfavorable_max_expectancy_pips", 0.0, epic))
    unfavorable_max_wr = float(_cfg(config, "epic_regime_health_unfavorable_max_win_rate", 0.45, epic))

    overall = _metrics(overall_rows)
    bucket = _metrics(bucket_rows)
    overall_state = _classify(
        overall,
        min_trades=min_trades,
        favorable_min_pf=favorable_min_pf,
        favorable_min_expectancy=favorable_min_exp,
        favorable_min_wr=favorable_min_wr,
        unfavorable_max_pf=unfavorable_max_pf,
        unfavorable_max_expectancy=unfavorable_max_exp,
        unfavorable_max_wr=unfavorable_max_wr,
    )
    bucket_state = _classify(
        bucket,
        min_trades=bucket_min_trades,
        favorable_min_pf=favorable_min_pf,
        favorable_min_expectancy=favorable_min_exp,
        favorable_min_wr=favorable_min_wr,
        unfavorable_max_pf=unfavorable_max_pf,
        unfavorable_max_expectancy=unfavorable_max_exp,
        unfavorable_max_wr=unfavorable_max_wr,
    )

    if overall_state == "unfavorable" or bucket_state == "unfavorable":
        state = "unfavorable"
    elif overall_state == "favorable" and bucket_state != "unfavorable":
        state = "favorable"
    else:
        state = "neutral"

    pf_text = "inf" if overall.profit_factor is None else f"{overall.profit_factor:.2f}"
    bucket_pf = "inf" if bucket.profit_factor is None else f"{bucket.profit_factor:.2f}"
    reason = (
        f"exec={execution_id} overall={overall_state} "
        f"n={overall.trades} WR={overall.win_rate:.0%} PF={pf_text} "
        f"exp={overall.expectancy:.2f}; bucket[{bucket_key}]={bucket_state} "
        f"n={bucket.trades} WR={bucket.win_rate:.0%} PF={bucket_pf} exp={bucket.expectancy:.2f}"
    )
    decision = HealthDecision(
        state=state,
        reason=reason,
        execution_id=execution_id,
        overall=overall,
        bucket=bucket,
        bucket_key=bucket_key,
    )
    _cache[cache_key] = (now, decision)
    return decision


def apply_epic_regime_health_gate(
    signal: Optional[Dict[str, Any]],
    config: Any,
    strategy_logger: Optional[logging.Logger] = None,
    *,
    signal_timestamp: Optional[datetime] = None,
    backtest_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    if signal is None:
        return None

    log = strategy_logger or logger
    epic = str(signal.get("epic", ""))
    enabled = _bool(_cfg(config, "epic_regime_health_enabled", False, epic))
    if not enabled:
        signal.setdefault("epic_regime_health_state", "disabled")
        return signal

    mode = str(_cfg(config, "epic_regime_health_mode", "MONITORING", epic)).upper()
    decision = evaluate_epic_regime_health(
        signal,
        config,
        signal_timestamp=signal_timestamp,
        backtest_mode=backtest_mode,
    )
    if decision is None:
        signal["epic_regime_health_state"] = "unknown"
        signal["epic_regime_health_reason"] = "health query failed"
        return signal

    signal["epic_regime_health_state"] = decision.state
    signal["epic_regime_health_reason"] = decision.reason
    signal["epic_regime_health_execution_id"] = decision.execution_id
    signal["epic_regime_health_bucket"] = decision.bucket_key
    signal["epic_regime_health_would_block"] = decision.state == "unfavorable"
    metadata = signal.setdefault("strategy_metadata", {})
    if isinstance(metadata, dict):
        metadata["epic_regime_health"] = {
            "state": decision.state,
            "reason": decision.reason,
            "execution_id": decision.execution_id,
            "bucket": decision.bucket_key,
            "would_block": decision.state == "unfavorable",
            "mode": mode,
        }

    prefix = "[EpicRegimeHealth]"
    if decision.state == "favorable":
        log.info("%s FAVORABLE %s", prefix, decision.reason)
    elif decision.state == "unfavorable":
        log.warning("%s UNFAVORABLE %s (%s mode)", prefix, decision.reason, mode)
        if mode == "ACTIVE":
            return None
    else:
        log.info("%s NEUTRAL %s", prefix, decision.reason)

    return signal
