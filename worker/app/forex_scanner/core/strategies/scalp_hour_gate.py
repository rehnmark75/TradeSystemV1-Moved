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
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

_cache: Dict[str, tuple[datetime, List[Dict[str, Any]]]] = {}
_CONDITION_FIELDS = {
    "market_regime_detected",
    "market_regime",
    "volatility_state",
    "atr_percentile",
    "bb_width_percentile",
    "efficiency_ratio",
    "entry_quality_score",
    "entry_candle_momentum",
    "mtf_confluence_score",
    "adx_value",
}


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
    bucket_mode: str
    bucket_key: str
    signal_direction: str
    condition_blocked: bool = False
    condition_rule: str = ""
    condition_context: Optional[Dict[str, Any]] = None

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
            "bucket_mode": self.bucket_mode,
            "bucket_key": self.bucket_key,
            "signal_direction": self.signal_direction,
            "condition_blocked": self.condition_blocked,
            "condition_rule": self.condition_rule,
            "condition_context": self.condition_context or {},
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


def _normalize_direction(direction: Optional[str]) -> str:
    return str(direction or "").upper()


def _csv_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_stats(
    sample_size: int,
    win_rate: float,
    profit_factor: Optional[float],
    expectancy: Optional[float],
) -> Dict[str, Any]:
    return {
        "sample_size": sample_size,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_pips": expectancy,
    }


def _row_context(row: Dict[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    nested = row.get("indicator_values")
    if isinstance(nested, dict):
        for key in _CONDITION_FIELDS:
            if nested.get(key) is not None:
                context[key] = nested.get(key)
    for key in _CONDITION_FIELDS:
        if row.get(key) is not None:
            context[key] = row.get(key)
    return context


def _condition_labels(context: Dict[str, Any], config: Any, epic: str) -> set[str]:
    labels: set[str] = set()
    atr_pct = _to_float(context.get("atr_percentile"))
    bb_pct = _to_float(context.get("bb_width_percentile"))
    regime = str(context.get("market_regime_detected") or context.get("market_regime") or "").lower()
    volatility_state = str(context.get("volatility_state") or "").lower()
    entry_quality = _to_float(context.get("entry_quality_score"))
    momentum = _to_float(context.get("entry_candle_momentum"))
    mtf = _to_float(context.get("mtf_confluence_score"))
    er = _to_float(context.get("efficiency_ratio"))

    compressed_atr = float(_cfg(config, "scalp_condition_hour_gate_compression_atr_max", 50.0, epic))
    compressed_bb = float(_cfg(config, "scalp_condition_hour_gate_compression_bb_max", 50.0, epic))
    expanded_pct = float(_cfg(config, "scalp_condition_hour_gate_expanded_percentile_min", 70.0, epic))
    low_quality_max = float(_cfg(config, "scalp_condition_hour_gate_low_quality_max", 0.50, epic))
    low_momentum_max = float(_cfg(config, "scalp_condition_hour_gate_low_momentum_max", 0.50, epic))
    low_confluence_max = float(_cfg(config, "scalp_condition_hour_gate_low_confluence_max", 0.35, epic))
    low_efficiency_max = float(_cfg(config, "scalp_condition_hour_gate_low_efficiency_max", 0.20, epic))

    if atr_pct is not None and bb_pct is not None and atr_pct < compressed_atr and bb_pct < compressed_bb:
        labels.add("vol_compressed")
    if (atr_pct is not None and atr_pct >= expanded_pct) or (bb_pct is not None and bb_pct >= expanded_pct):
        labels.add("vol_expanded")
    if regime == "high_volatility" and volatility_state == "extreme":
        labels.add("regime_extreme")
    if entry_quality is not None and entry_quality < low_quality_max:
        labels.add("low_quality")
    if momentum is not None and momentum < low_momentum_max:
        labels.add("low_momentum")
    if mtf is not None and mtf < low_confluence_max:
        labels.add("low_confluence")
    if er is not None and er < low_efficiency_max:
        labels.add("low_efficiency")
    return labels


def _bucket_key(hour_utc: int, signal_direction: str, bucket_mode: str) -> str:
    direction = _normalize_direction(signal_direction) or "UNKNOWN"
    mode = (bucket_mode or "hour").lower()
    if mode == "direction":
        return direction
    if mode == "hour4":
        start = (int(hour_utc) // 4) * 4
        return f"h{start:02d}_{start + 3:02d}"
    if mode == "direction_hour":
        return f"{direction}:h{int(hour_utc):02d}"
    if mode == "direction_hour4":
        start = (int(hour_utc) // 4) * 4
        return f"{direction}:h{start:02d}_{start + 3:02d}"
    return f"h{int(hour_utc):02d}"


def _hour_bounds(hour_utc: int, bucket_mode: str) -> tuple[int, int]:
    mode = (bucket_mode or "hour").lower()
    if mode in {"hour4", "direction_hour4"}:
        start = (int(hour_utc) // 4) * 4
        return start, start + 3
    return int(hour_utc), int(hour_utc)


def _uses_direction(bucket_mode: str) -> bool:
    return (bucket_mode or "hour").lower() in {"direction", "direction_hour", "direction_hour4"}


def _deterministic_probe(
    epic: str,
    hour_utc: int,
    bucket_key: str,
    timestamp: Optional[datetime],
    rate: float,
) -> bool:
    if rate <= 0:
        return False
    day = (timestamp or datetime.utcnow()).date()
    raw = f"{epic}:{hour_utc}:{bucket_key}:{day}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < rate


def _load_live_rows(
    epic: str,
    hour_utc: int,
    signal_direction: str,
    bucket_mode: str,
    lookback_days: int,
    window: int,
    environment: str,
    cache_ttl_seconds: int,
) -> List[Dict[str, Any]]:
    now = datetime.utcnow()
    bucket_key = _bucket_key(hour_utc, signal_direction, bucket_mode)
    hour_start, hour_end = _hour_bounds(hour_utc, bucket_mode)
    direction_filter = _uses_direction(bucket_mode)
    direction = _normalize_direction(signal_direction)
    cache_key = f"live:{environment}:{epic}:{bucket_mode}:{bucket_key}:{lookback_days}:{window}"
    cached = _cache.get(cache_key)
    if cached and (now - cached[0]).total_seconds() < cache_ttl_seconds:
        return cached[1]

    query = """
        SELECT
            t.closed_at,
            t.pips_gained,
            t.profit_loss,
            a.market_regime_detected,
            a.market_regime,
            a.volatility_state,
            a.atr_percentile,
            a.bb_width_percentile,
            a.efficiency_ratio,
            a.entry_quality_score,
            a.entry_candle_momentum,
            a.mtf_confluence_score,
            a.adx_value
        FROM trade_log t
        JOIN alert_history a ON a.id = t.alert_id
        WHERE t.symbol = %s
          AND a.strategy = 'SMC_SIMPLE'
          AND t.status = 'closed'
          AND t.closed_at IS NOT NULL
          AND t.closed_at >= NOW() - (%s::text || ' days')::interval
          AND extract(hour from COALESCE(a.alert_timestamp, t.closed_at)) BETWEEN %s AND %s
          AND (%s = false OR a.signal_type = %s)
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
                cur.execute(
                    query,
                    [
                        epic,
                        lookback_days,
                        hour_start,
                        hour_end,
                        direction_filter,
                        direction,
                        environment,
                        environment,
                        window,
                    ],
                )
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
    signal_direction: str,
    bucket_mode: str,
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
    bucket_key = _bucket_key(hour_utc, signal_direction, bucket_mode)
    hour_start, hour_end = _hour_bounds(hour_utc, bucket_mode)
    direction_filter = _uses_direction(bucket_mode)
    direction = _normalize_direction(signal_direction)
    cache_key = f"seed:{execution_id}:{strategy_name}:{epic}:{bucket_mode}:{bucket_key}:{before_ts.isoformat()}:{window}"
    cached = _cache.get(cache_key)
    if cached and (datetime.utcnow() - cached[0]).total_seconds() < cache_ttl_seconds:
        return cached[1]

    query = """
        SELECT
            signal_timestamp AS closed_at,
            pips_gained,
            NULL::numeric AS profit_loss,
            indicator_values,
            indicator_values->>'market_regime_detected' AS market_regime_detected,
            indicator_values->>'market_regime' AS market_regime,
            indicator_values->>'volatility_state' AS volatility_state,
            indicator_values->>'atr_percentile' AS atr_percentile,
            indicator_values->>'bb_width_percentile' AS bb_width_percentile,
            indicator_values->>'efficiency_ratio' AS efficiency_ratio,
            indicator_values->>'entry_quality_score' AS entry_quality_score,
            indicator_values->>'entry_candle_momentum' AS entry_candle_momentum,
            indicator_values->>'mtf_confluence_score' AS mtf_confluence_score,
            indicator_values->>'adx_value' AS adx_value
        FROM backtest_signals
        WHERE execution_id = %s
          AND epic = %s
          AND strategy_name = %s
          AND extract(hour from signal_timestamp) BETWEEN %s AND %s
          AND (%s = false OR signal_type = %s)
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
                cur.execute(
                    query,
                    [
                        int(execution_id),
                        epic,
                        strategy_name,
                        hour_start,
                        hour_end,
                        direction_filter,
                        direction,
                        before_ts,
                        window,
                    ],
                )
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


def _condition_block(
    *,
    epic: str,
    bucket_key: str,
    config: Any,
    condition_context: Optional[Dict[str, Any]],
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    if not _bool(_cfg(config, "scalp_condition_hour_gate_enabled", True, epic)):
        return False, "", {}

    target_epics = _csv_values(_cfg(config, "scalp_condition_hour_gate_target_epics", "CS.D.EURUSD.CEEM.IP", epic))
    if target_epics and epic not in target_epics:
        return False, "", {}

    context = condition_context or {}
    atr_pct = _to_float(context.get("atr_percentile"))
    bb_pct = _to_float(context.get("bb_width_percentile"))
    regime = str(context.get("market_regime_detected") or context.get("market_regime") or "").lower()
    volatility_state = str(context.get("volatility_state") or "").lower()
    current_labels = _condition_labels(context, config, epic)

    dynamic_enabled = _bool(_cfg(config, "scalp_condition_hour_gate_dynamic_enabled", True, epic))
    if dynamic_enabled and current_labels and rows:
        min_trades = int(_cfg(config, "scalp_condition_hour_gate_dynamic_min_trades", 8, epic))
        block_pf = float(_cfg(config, "scalp_condition_hour_gate_dynamic_block_profit_factor", 0.75, epic))
        block_expectancy = float(
            _cfg(config, "scalp_condition_hour_gate_dynamic_block_expectancy_pips", -0.25, epic)
        )
        candidate_blocks: List[Tuple[str, int, float, Optional[float], Optional[float]]] = []
        for label in sorted(current_labels):
            label_rows = [
                row
                for row in rows
                if label in _condition_labels(_row_context(row), config, epic)
            ]
            sample_size, win_rate, profit_factor, expectancy = _stats(label_rows)
            if (
                sample_size >= min_trades
                and profit_factor is not None
                and expectancy is not None
                and profit_factor < block_pf
                and expectancy < block_expectancy
            ):
                candidate_blocks.append((label, sample_size, win_rate, profit_factor, expectancy))

        if candidate_blocks:
            label, sample_size, win_rate, profit_factor, expectancy = sorted(
                candidate_blocks,
                key=lambda item: (item[4] if item[4] is not None else 0.0),
            )[0]
            return True, f"dynamic_condition:{label}", {
                "condition_label": label,
                "current_labels": sorted(current_labels),
                **_compact_stats(sample_size, win_rate, profit_factor, expectancy),
                "block_profit_factor": block_pf,
                "block_expectancy_pips": block_expectancy,
            }

    fallback_enabled = _bool(_cfg(config, "scalp_condition_hour_gate_static_fallback_enabled", True, epic))
    if not fallback_enabled:
        return False, "", {}

    compressed_buckets = _csv_values(
        _cfg(config, "scalp_condition_hour_gate_compression_buckets", "BULL:h00_03,BULL:h08_11", epic)
    )
    atr_max = float(_cfg(config, "scalp_condition_hour_gate_compression_atr_max", 50.0, epic))
    bb_max = float(_cfg(config, "scalp_condition_hour_gate_compression_bb_max", 50.0, epic))
    if (
        bucket_key in compressed_buckets
        and atr_pct is not None
        and bb_pct is not None
        and atr_pct < atr_max
        and bb_pct < bb_max
    ):
        return True, "compressed_bull_hour_bucket", {
            "atr_percentile": atr_pct,
            "bb_width_percentile": bb_pct,
            "atr_max": atr_max,
            "bb_max": bb_max,
        }

    extreme_buckets = _csv_values(
        _cfg(config, "scalp_condition_hour_gate_extreme_buckets", "BEAR:h08_11", epic)
    )
    extreme_regime = str(
        _cfg(config, "scalp_condition_hour_gate_extreme_regime", "high_volatility", epic)
    ).lower()
    extreme_volatility = str(
        _cfg(config, "scalp_condition_hour_gate_extreme_volatility_state", "extreme", epic)
    ).lower()
    if bucket_key in extreme_buckets and regime == extreme_regime and volatility_state == extreme_volatility:
        return True, "extreme_volatility_bear_hour_bucket", {
            "market_regime_detected": regime,
            "volatility_state": volatility_state,
        }

    return False, "", {}


def evaluate_scalp_hour_gate(
    *,
    epic: str,
    pair: str,
    hour_utc: int,
    signal_direction: Optional[str] = None,
    static_blocked_hours: Optional[List[int]],
    config: Any,
    signal_timestamp: Optional[datetime],
    backtest_mode: bool,
    environment: Optional[str] = None,
    condition_context: Optional[Dict[str, Any]] = None,
) -> HourGateDecision:
    static_blocked = hour_utc in set(static_blocked_hours or [])
    enabled = _bool(_cfg(config, "scalp_hour_gate_enabled", True, epic))
    mode = str(_cfg(config, "scalp_hour_gate_mode", "ACTIVE", epic)).upper()
    bucket_mode = str(_cfg(config, "scalp_hour_gate_bucket_mode", "hour", epic)).lower()
    if bucket_mode not in {"hour", "hour4", "direction", "direction_hour", "direction_hour4"}:
        logger.warning("[ScalpHourGate] unknown bucket mode '%s'; falling back to hour", bucket_mode)
        bucket_mode = "hour"
    direction = _normalize_direction(signal_direction)
    gate_bucket = _bucket_key(hour_utc, direction, bucket_mode)
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
            bucket_mode=bucket_mode,
            bucket_key=gate_bucket,
            signal_direction=direction,
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
        rows = _load_seed_rows(
            epic, hour_utc, direction, bucket_mode, config, signal_timestamp, window, cache_ttl_seconds
        )
        stats_source = "seed_backtest" if rows else "backtest_no_seed"
    else:
        rows = _load_live_rows(
            epic, hour_utc, direction, bucket_mode, lookback_days, window, env, cache_ttl_seconds
        )
        stats_source = "live"
        if len(rows) < min_trades:
            seed_rows = _load_seed_rows(
                epic, hour_utc, direction, bucket_mode, config, signal_timestamp, window, cache_ttl_seconds
            )
            if seed_rows:
                rows = (rows + seed_rows)[:window]
                stats_source = "live_seed_backtest"

    sample_size, win_rate, profit_factor, expectancy = _stats(rows)
    has_evidence = sample_size >= min_trades and profit_factor is not None and expectancy is not None
    dynamic_blocked = has_evidence and profit_factor < block_pf and expectancy < block_expectancy
    dynamic_released = has_evidence and profit_factor >= release_pf and expectancy >= release_expectancy
    condition_blocked, condition_rule, matched_condition_context = _condition_block(
        epic=epic,
        bucket_key=gate_bucket,
        config=config,
        condition_context=condition_context,
        rows=rows,
    )
    probe = False

    condition_mode = str(_cfg(config, "scalp_condition_hour_gate_mode", "ACTIVE", epic)).upper()
    if condition_blocked:
        allowed = condition_mode != "ACTIVE"
        state = "condition_block"
    elif dynamic_blocked:
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
        probe = _deterministic_probe(epic, hour_utc, gate_bucket, signal_timestamp, probe_rate)
        allowed = probe or mode != "ACTIVE"
        state = "static_probe" if probe else "static_block"
    else:
        allowed = True
        state = "open"

    pf_text = "n/a" if profit_factor is None else f"{profit_factor:.2f}"
    exp_text = "n/a" if expectancy is None else f"{expectancy:.2f}"
    reason = (
        f"Scalp hour gate {state} for {pair} bucket={gate_bucket} hour={hour_utc} UTC "
        f"(static={static_blocked}, n={sample_size}, wr={win_rate:.0%}, "
        f"pf={pf_text}, expectancy={exp_text}, source={stats_source}, mode={bucket_mode})"
    )
    if condition_blocked:
        reason += f" condition_rule={condition_rule}"
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
        bucket_mode=bucket_mode,
        bucket_key=gate_bucket,
        signal_direction=direction,
        condition_blocked=condition_blocked,
        condition_rule=condition_rule,
        condition_context=matched_condition_context,
    )
