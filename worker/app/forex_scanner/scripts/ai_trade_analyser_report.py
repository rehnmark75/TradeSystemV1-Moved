#!/usr/bin/env python3
"""
AI Trade Analyser report generator.

Builds a markdown report comparing Claude-approved trades against
Claude-rejected signal outcomes for a recent analysis window.

Intended to run inside the task-worker container:

    docker exec -it task-worker python /app/forex_scanner/scripts/ai_trade_analyser_report.py --days 14
"""

from __future__ import annotations

import argparse
import os
import struct
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import psycopg2
import psycopg2.extras


DEFAULT_DSN = os.getenv(
    "TRADING_DSN",
    "host=postgres dbname=forex user=postgres password=postgres",
)
DEFAULT_CONFIG_DSN = os.getenv(
    "STRATEGY_CONFIG_DSN",
    "host=postgres dbname=strategy_config user=postgres password=postgres",
)
DEFAULT_TIMEOUT_HOURS = 48


@dataclass
class Recommendation:
    category: str
    severity: str
    message: str
    evidence: str


@dataclass
class ChartArtifact:
    alert_id: int
    pair: str
    decision: str
    chart_url: str
    storage_type: str
    fetched: bool
    bytes_size: int
    width: int | None
    height: int | None
    linked_result_found: bool
    linked_prompt_found: bool
    result_reason: str


@dataclass
class SimulatedOutcome:
    alert_id: int
    epic: str
    pair: str
    signal_type: str
    market_session: str | None
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pips: float
    take_profit_pips: float
    outcome: str
    pips: float
    minutes_to_outcome: int | None
    max_favorable_excursion_pips: float
    max_adverse_excursion_pips: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI trade analysis report")
    parser.add_argument("--days", type=int, default=14, help="Lookback window in days")
    parser.add_argument("--pair", default=None, help="Optional pair filter, e.g. EURUSD")
    parser.add_argument("--strategy", default=None, help="Optional strategy filter")
    parser.add_argument("--environment", default="demo", help="Environment filter")
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help="psycopg2 DSN for forex/trading database",
    )
    parser.add_argument(
        "--config-dsn",
        default=DEFAULT_CONFIG_DSN,
        help="psycopg2 DSN for strategy_config database",
    )
    parser.add_argument(
        "--timeout-hours",
        type=int,
        default=DEFAULT_TIMEOUT_HOURS,
        help="Maximum candle walk horizon for rejected-alert outcome simulation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path; defaults to stdout",
    )
    return parser.parse_args()


def fetch_one(cur, query: str, params: list[Any]) -> dict[str, Any]:
    cur.execute(query, params)
    row = cur.fetchone()
    return dict(row) if row else {}


def fetch_all(cur, query: str, params: list[Any]) -> list[dict[str, Any]]:
    cur.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def compact_reason(reason: str | None) -> str:
    if not reason:
        return "No recorded reason"
    cleaned = " ".join(str(reason).split())
    for delimiter in (". ", "\n", "; "):
        if delimiter in cleaned:
            cleaned = cleaned.split(delimiter)[0]
            break
    return cleaned[:140]


def pip_size(epic: str) -> float:
    epic_upper = (epic or "").upper()
    if "CFEGOLD" in epic_upper or "GOLD" in epic_upper:
        return 0.1
    if "JPY" in epic_upper:
        return 0.01
    return 0.0001


def parse_png_dimensions(payload: bytes) -> tuple[int | None, int | None]:
    if len(payload) < 24:
        return None, None
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    try:
        width, height = struct.unpack(">II", payload[16:24])
        return int(width), int(height)
    except struct.error:
        return None, None


def chart_storage_type(chart_url: str) -> str:
    if chart_url.startswith("file://"):
        return "file"
    if chart_url.startswith("http://") or chart_url.startswith("https://"):
        return "url"
    return "path"


def fetch_chart_bytes(chart_url: str, timeout: int = 10) -> bytes | None:
    try:
        if chart_url.startswith("file://"):
            path = chart_url.replace("file://", "", 1)
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    return handle.read()
            return None

        if chart_url.startswith("http://") or chart_url.startswith("https://"):
            with urllib.request.urlopen(chart_url, timeout=timeout) as response:
                return response.read()

        if os.path.exists(chart_url):
            with open(chart_url, "rb") as handle:
                return handle.read()
    except (OSError, urllib.error.URLError, ValueError):
        return None

    return None


def derive_artifact_prefix(chart_url: str) -> str | None:
    basename = os.path.basename(chart_url.replace("file://", "", 1))
    if basename.endswith("_chart.png"):
        return basename[:-10]
    if basename.endswith(".png"):
        return basename[:-4]
    return None


def read_text_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def inspect_chart_artifact(row: dict[str, Any]) -> ChartArtifact:
    chart_url = str(row.get("vision_chart_url") or "")
    decision = (
        "APPROVE"
        if row.get("claude_approved") is True or row.get("claude_decision") == "APPROVE"
        else "REJECT"
    )
    payload = fetch_chart_bytes(chart_url) if chart_url else None
    width, height = parse_png_dimensions(payload) if payload else (None, None)
    prefix = derive_artifact_prefix(chart_url) if chart_url else None

    result_found = False
    prompt_found = False
    result_reason = ""
    if prefix:
        artifact_dir = "/app/claude_analysis_enhanced/vision_analysis"
        result_path = os.path.join(artifact_dir, f"{prefix}_result.json")
        prompt_path = os.path.join(artifact_dir, f"{prefix}_prompt.txt")
        result_found = os.path.exists(result_path)
        prompt_found = os.path.exists(prompt_path)
        if result_found:
            result_text = read_text_file(result_path)
            if result_text:
                result_reason = compact_reason(result_text)

    return ChartArtifact(
        alert_id=int(row.get("id") or 0),
        pair=str(row.get("pair") or ""),
        decision=decision,
        chart_url=chart_url,
        storage_type=chart_storage_type(chart_url) if chart_url else "missing",
        fetched=payload is not None,
        bytes_size=len(payload) if payload else 0,
        width=width,
        height=height,
        linked_result_found=result_found,
        linked_prompt_found=prompt_found,
        result_reason=result_reason,
    )


def build_filters(
    table_alias: str,
    days: int,
    pair: str | None,
    strategy: str | None,
    environment: str | None,
    timestamp_column: str = "alert_timestamp",
) -> tuple[str, list[Any]]:
    clauses = [f"{table_alias}.{timestamp_column} >= NOW() - INTERVAL %s"]
    params: list[Any] = [f"{days} days"]

    if pair:
        clauses.append(f"{table_alias}.pair = %s")
        params.append(pair)

    if strategy:
        clauses.append(f"{table_alias}.strategy = %s")
        params.append(strategy)

    if environment:
        clauses.append(f"{table_alias}.environment = %s")
        params.append(environment)

    return " AND ".join(clauses), params


def get_alert_stats(cur, args: argparse.Namespace) -> dict[str, Any]:
    where_sql, params = build_filters("a", args.days, args.pair, args.strategy, args.environment)
    query = f"""
    SELECT
        COUNT(*) AS total_alerts,
        COUNT(*) FILTER (
            WHERE a.claude_approved = TRUE OR a.claude_decision = 'APPROVE'
        ) AS approved_alerts,
        COUNT(*) FILTER (
            WHERE a.claude_approved = FALSE
               OR a.claude_decision = 'REJECT'
               OR a.alert_level = 'REJECTED'
        ) AS rejected_alerts,
        ROUND(AVG(a.claude_score)::numeric, 2) AS avg_claude_score
    FROM alert_history a
    WHERE {where_sql}
    """
    return fetch_one(cur, query, params)


def get_approved_trade_summary(cur, args: argparse.Namespace) -> dict[str, Any]:
    where_sql, params = build_filters("a", args.days, args.pair, args.strategy, args.environment)
    query = f"""
    WITH approved_trades AS (
        SELECT
            a.id AS alert_id,
            a.pair,
            a.market_session,
            a.signal_type,
            a.claude_score,
            COALESCE(t.pips_gained, 0) AS pips_gained,
            t.profit_loss,
            t.moved_to_breakeven,
            t.partial_close_executed,
            CASE
                WHEN COALESCE(t.pips_gained, 0) > 0 THEN 'WIN'
                WHEN COALESCE(t.pips_gained, 0) < 0 THEN 'LOSS'
                WHEN t.profit_loss > 0 THEN 'WIN'
                WHEN t.profit_loss < 0 THEN 'LOSS'
                ELSE 'BREAKEVEN'
            END AS trade_result
        FROM alert_history a
        JOIN trade_log t ON t.alert_id = a.id
        WHERE {where_sql}
          AND (a.claude_approved = TRUE OR a.claude_decision = 'APPROVE')
          AND t.closed_at IS NOT NULL
    )
    SELECT
        COUNT(*) AS total_closed_trades,
        COUNT(*) FILTER (WHERE trade_result = 'WIN') AS wins,
        COUNT(*) FILTER (WHERE trade_result = 'LOSS') AS losses,
        COUNT(*) FILTER (WHERE trade_result = 'BREAKEVEN') AS breakevens,
        ROUND(
            COUNT(*) FILTER (WHERE trade_result = 'WIN')::numeric /
            NULLIF(COUNT(*) FILTER (WHERE trade_result IN ('WIN', 'LOSS')), 0) * 100,
            1
        ) AS win_rate,
        ROUND(COALESCE(SUM(pips_gained), 0)::numeric, 1) AS net_pips,
        ROUND(COALESCE(AVG(pips_gained), 0)::numeric, 2) AS avg_pips,
        ROUND(AVG(claude_score)::numeric, 2) AS avg_claude_score,
        COUNT(*) FILTER (WHERE moved_to_breakeven = TRUE) AS moved_to_breakeven_count,
        COUNT(*) FILTER (WHERE partial_close_executed = TRUE) AS partial_close_count
    FROM approved_trades
    """
    return fetch_one(cur, query, params)


def get_approved_pair_breakdown(cur, args: argparse.Namespace) -> list[dict[str, Any]]:
    where_sql, params = build_filters("a", args.days, args.pair, args.strategy, args.environment)
    query = f"""
    WITH approved_trades AS (
        SELECT
            a.pair,
            COALESCE(t.pips_gained, 0) AS pips_gained,
            CASE
                WHEN COALESCE(t.pips_gained, 0) > 0 THEN 'WIN'
                WHEN COALESCE(t.pips_gained, 0) < 0 THEN 'LOSS'
                WHEN t.profit_loss > 0 THEN 'WIN'
                WHEN t.profit_loss < 0 THEN 'LOSS'
                ELSE 'BREAKEVEN'
            END AS trade_result
        FROM alert_history a
        JOIN trade_log t ON t.alert_id = a.id
        WHERE {where_sql}
          AND (a.claude_approved = TRUE OR a.claude_decision = 'APPROVE')
          AND t.closed_at IS NOT NULL
    )
    SELECT
        pair,
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE trade_result = 'WIN') AS wins,
        COUNT(*) FILTER (WHERE trade_result = 'LOSS') AS losses,
        ROUND(
            COUNT(*) FILTER (WHERE trade_result = 'WIN')::numeric /
            NULLIF(COUNT(*) FILTER (WHERE trade_result IN ('WIN', 'LOSS')), 0) * 100,
            1
        ) AS win_rate,
        ROUND(SUM(pips_gained)::numeric, 1) AS net_pips
    FROM approved_trades
    GROUP BY pair
    HAVING COUNT(*) >= 3
    ORDER BY net_pips DESC, total DESC
    LIMIT 8
    """
    return fetch_all(cur, query, params)


def get_rejected_alert_reasons(cur, args: argparse.Namespace) -> list[dict[str, Any]]:
    where_sql, params = build_filters("a", args.days, args.pair, args.strategy, args.environment)
    query = f"""
    SELECT
        a.claude_reason,
        COUNT(*) AS total
    FROM alert_history a
    WHERE {where_sql}
      AND (
        a.claude_approved = FALSE
        OR a.claude_decision = 'REJECT'
        OR a.alert_level = 'REJECTED'
      )
    GROUP BY a.claude_reason
    ORDER BY total DESC
    LIMIT 12
    """
    rows = fetch_all(cur, query, params)
    normalized = Counter()
    for row in rows:
        normalized[compact_reason(row.get("claude_reason"))] += int(row.get("total") or 0)
    return [
        {"reason": reason, "total": total}
        for reason, total in normalized.most_common(8)
    ]


def get_chart_rows(cur, args: argparse.Namespace) -> list[dict[str, Any]]:
    where_sql, params = build_filters("a", args.days, args.pair, args.strategy, args.environment)
    query = f"""
    SELECT
        a.id,
        a.pair,
        a.claude_decision,
        a.claude_approved,
        a.vision_chart_url
    FROM alert_history a
    WHERE {where_sql}
      AND a.vision_chart_url IS NOT NULL
      AND a.vision_chart_url <> ''
    ORDER BY a.alert_timestamp DESC
    LIMIT 120
    """
    return fetch_all(cur, query, params)


def load_pair_risk_config(config_dsn: str) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    pair_config: dict[str, tuple[float, float]] = {}
    global_sl = 9.0
    global_tp = 15.0

    with psycopg2.connect(config_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT fixed_stop_loss_pips, fixed_take_profit_pips
                FROM smc_simple_global_config
                WHERE is_active = TRUE
                ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if row:
                if row.get("fixed_stop_loss_pips") is not None:
                    global_sl = float(row["fixed_stop_loss_pips"])
                if row.get("fixed_take_profit_pips") is not None:
                    global_tp = float(row["fixed_take_profit_pips"])

            cur.execute(
                """
                SELECT epic, fixed_stop_loss_pips, fixed_take_profit_pips
                FROM smc_simple_pair_overrides
                WHERE is_enabled = TRUE OR is_enabled IS NULL
                """
            )
            for row in cur.fetchall():
                epic = row.get("epic")
                if not epic:
                    continue
                sl = float(row["fixed_stop_loss_pips"]) if row.get("fixed_stop_loss_pips") is not None else global_sl
                tp = float(row["fixed_take_profit_pips"]) if row.get("fixed_take_profit_pips") is not None else global_tp
                pair_config[epic] = (sl, tp)

    return pair_config, (global_sl, global_tp)


def get_rejected_alert_rows(cur, args: argparse.Namespace) -> list[dict[str, Any]]:
    where_sql, params = build_filters("a", args.days, args.pair, args.strategy, args.environment)
    query = f"""
    SELECT
        a.id,
        a.alert_timestamp,
        a.epic,
        a.pair,
        a.signal_type,
        a.price,
        a.confidence_score,
        a.claude_score,
        a.claude_reason,
        a.market_session
    FROM alert_history a
    WHERE {where_sql}
      AND (
        a.claude_approved = FALSE
        OR a.claude_decision = 'REJECT'
        OR a.alert_level = 'REJECTED'
      )
    ORDER BY a.alert_timestamp DESC
    LIMIT 500
    """
    return fetch_all(cur, query, params)


def simulate_rejected_alert_outcome(
    cur,
    row: dict[str, Any],
    pair_config: dict[str, tuple[float, float]],
    global_config: tuple[float, float],
    timeout_hours: int,
) -> SimulatedOutcome:
    epic = str(row.get("epic") or "")
    signal_type = str(row.get("signal_type") or "")
    entry_price = float(row.get("price") or 0.0)
    ts = row.get("alert_timestamp")
    sl_pips, tp_pips = pair_config.get(epic, global_config)
    pip = pip_size(epic)
    is_long = signal_type.upper() in ("BULL", "BUY", "LONG")

    if is_long:
        sl_price = entry_price - sl_pips * pip
        tp_price = entry_price + tp_pips * pip
    else:
        sl_price = entry_price + sl_pips * pip
        tp_price = entry_price - tp_pips * pip

    end_ts = ts + timedelta(hours=timeout_hours)
    cur.execute(
        """
        SELECT start_time, high, low
        FROM ig_candles
        WHERE epic = %s AND timeframe = 1
          AND start_time >= %s AND start_time <= %s
        ORDER BY start_time ASC
        """,
        (epic, ts, end_ts),
    )
    candles = cur.fetchall()
    if not candles:
        cur.execute(
            """
            SELECT start_time, high, low
            FROM ig_candles
            WHERE epic = %s AND timeframe = 5
              AND start_time >= %s AND start_time <= %s
            ORDER BY start_time ASC
            """,
            (epic, ts, end_ts),
        )
        candles = cur.fetchall()

    if not candles:
        return SimulatedOutcome(
            alert_id=int(row["id"]),
            epic=epic,
            pair=str(row.get("pair") or ""),
            signal_type=signal_type,
            market_session=row.get("market_session"),
            entry_price=entry_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            stop_loss_pips=sl_pips,
            take_profit_pips=tp_pips,
            outcome="NO_DATA",
            pips=0.0,
            minutes_to_outcome=None,
            max_favorable_excursion_pips=0.0,
            max_adverse_excursion_pips=0.0,
        )

    mfe = 0.0
    mae = 0.0
    for candle in candles:
        high = float(candle["high"])
        low = float(candle["low"])
        if is_long:
            favorable = (high - entry_price) / pip
            adverse = (entry_price - low) / pip
            if low <= sl_price and high >= tp_price:
                outcome, pips = "LOSS", -sl_pips
            elif low <= sl_price:
                outcome, pips = "LOSS", -sl_pips
            elif high >= tp_price:
                outcome, pips = "WIN", tp_pips
            else:
                outcome, pips = "", 0.0
        else:
            favorable = (entry_price - low) / pip
            adverse = (high - entry_price) / pip
            if high >= sl_price and low <= tp_price:
                outcome, pips = "LOSS", -sl_pips
            elif high >= sl_price:
                outcome, pips = "LOSS", -sl_pips
            elif low <= tp_price:
                outcome, pips = "WIN", tp_pips
            else:
                outcome, pips = "", 0.0

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

        if outcome:
            minutes = int((candle["start_time"] - ts).total_seconds() / 60)
            return SimulatedOutcome(
                alert_id=int(row["id"]),
                epic=epic,
                pair=str(row.get("pair") or ""),
                signal_type=signal_type,
                market_session=row.get("market_session"),
                entry_price=entry_price,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                stop_loss_pips=sl_pips,
                take_profit_pips=tp_pips,
                outcome=outcome,
                pips=pips,
                minutes_to_outcome=minutes,
                max_favorable_excursion_pips=round(mfe, 2),
                max_adverse_excursion_pips=round(mae, 2),
            )

    return SimulatedOutcome(
        alert_id=int(row["id"]),
        epic=epic,
        pair=str(row.get("pair") or ""),
        signal_type=signal_type,
        market_session=row.get("market_session"),
        entry_price=entry_price,
        stop_loss_price=sl_price,
        take_profit_price=tp_price,
        stop_loss_pips=sl_pips,
        take_profit_pips=tp_pips,
        outcome="TIMEOUT",
        pips=0.0,
        minutes_to_outcome=None,
        max_favorable_excursion_pips=round(mfe, 2),
        max_adverse_excursion_pips=round(mae, 2),
    )


def simulate_rejected_alerts(
    cur,
    args: argparse.Namespace,
    pair_config: dict[str, tuple[float, float]],
    global_config: tuple[float, float],
) -> list[SimulatedOutcome]:
    rows = get_rejected_alert_rows(cur, args)
    return [
        simulate_rejected_alert_outcome(cur, row, pair_config, global_config, args.timeout_hours)
        for row in rows
    ]


def build_simulated_rejection_summary(outcomes: list[SimulatedOutcome]) -> dict[str, Any]:
    winners = [o for o in outcomes if o.outcome == "WIN"]
    losers = [o for o in outcomes if o.outcome == "LOSS"]
    resolved = winners + losers
    timeouts = [o for o in outcomes if o.outcome == "TIMEOUT"]
    no_data = [o for o in outcomes if o.outcome == "NO_DATA"]
    return {
        "total_analyzed": len(outcomes),
        "would_be_winners": len(winners),
        "would_be_losers": len(losers),
        "still_open": len(timeouts),
        "insufficient_data": len(no_data),
        "win_rate": round((len(winners) / len(resolved) * 100), 1) if resolved else 0.0,
        "net_pips": round(sum(o.pips for o in resolved), 1),
        "avg_mfe": round(sum(o.max_favorable_excursion_pips for o in outcomes) / len(outcomes), 2) if outcomes else 0.0,
        "avg_mae": round(sum(o.max_adverse_excursion_pips for o in outcomes) / len(outcomes), 2) if outcomes else 0.0,
        "simulation_source": "ig_candles_direct",
    }


def build_simulated_pair_breakdown(outcomes: list[SimulatedOutcome]) -> list[dict[str, Any]]:
    buckets: dict[str, list[SimulatedOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.pair or outcome.epic, []).append(outcome)
    rows = []
    for pair, pair_outcomes in buckets.items():
        winners = [o for o in pair_outcomes if o.outcome == "WIN"]
        losers = [o for o in pair_outcomes if o.outcome == "LOSS"]
        resolved = winners + losers
        rows.append({
            "pair": pair,
            "total": len(pair_outcomes),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round((len(winners) / len(resolved) * 100), 1) if resolved else 0.0,
            "net_pips": round(sum(o.pips for o in resolved), 1),
        })
    return sorted(rows, key=lambda row: (row["net_pips"], row["total"]), reverse=True)[:8]


def build_simulated_session_breakdown(outcomes: list[SimulatedOutcome]) -> list[dict[str, Any]]:
    buckets: dict[str, list[SimulatedOutcome]] = {}
    for outcome in outcomes:
        session = outcome.market_session or "UNKNOWN"
        buckets.setdefault(session, []).append(outcome)
    rows = []
    for session, session_outcomes in buckets.items():
        winners = [o for o in session_outcomes if o.outcome == "WIN"]
        losers = [o for o in session_outcomes if o.outcome == "LOSS"]
        resolved = winners + losers
        rows.append({
            "market_session": session,
            "total": len(session_outcomes),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round((len(winners) / len(resolved) * 100), 1) if resolved else 0.0,
            "net_pips": round(sum(o.pips for o in resolved), 1),
        })
    return sorted(rows, key=lambda row: (row["net_pips"], row["total"]), reverse=True)[:6]


def summarize_chart_artifacts(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[ChartArtifact]]:
    artifacts = [inspect_chart_artifact(row) for row in rows]
    total = len(artifacts)
    fetched = [a for a in artifacts if a.fetched]
    approved = [a for a in artifacts if a.decision == "APPROVE"]
    rejected = [a for a in artifacts if a.decision == "REJECT"]
    file_backed = [a for a in artifacts if a.storage_type == "file"]
    url_backed = [a for a in artifacts if a.storage_type == "url"]
    linked_results = [a for a in artifacts if a.linked_result_found]
    linked_prompts = [a for a in artifacts if a.linked_prompt_found]

    dimensions = [(a.width, a.height) for a in fetched if a.width and a.height]
    common_dimension = Counter(dimensions).most_common(1)
    common_dimension_text = (
        f"{common_dimension[0][0][0]}x{common_dimension[0][0][1]}"
        if common_dimension
        else "Unknown"
    )

    summary = {
        "total_chart_alerts": total,
        "fetched_chart_alerts": len(fetched),
        "approved_chart_alerts": len(approved),
        "rejected_chart_alerts": len(rejected),
        "file_backed": len(file_backed),
        "url_backed": len(url_backed),
        "linked_results": len(linked_results),
        "linked_prompts": len(linked_prompts),
        "avg_chart_bytes": round(sum(a.bytes_size for a in fetched) / len(fetched), 1) if fetched else 0.0,
        "common_dimensions": common_dimension_text,
    }
    return summary, artifacts


def get_rejection_outcome_summary(cur, args: argparse.Namespace) -> dict[str, Any]:
    clauses = ["analysis_timestamp >= NOW() - INTERVAL %s"]
    params: list[Any] = [f"{args.days} days"]
    if args.pair:
        clauses.append("pair = %s")
        params.append(args.pair)

    where_sql = " AND ".join(clauses)
    query = f"""
    SELECT
        COUNT(*) AS total_analyzed,
        COUNT(*) FILTER (WHERE outcome = 'HIT_TP') AS would_be_winners,
        COUNT(*) FILTER (WHERE outcome = 'HIT_SL') AS would_be_losers,
        COUNT(*) FILTER (WHERE outcome = 'STILL_OPEN') AS still_open,
        COUNT(*) FILTER (WHERE outcome = 'INSUFFICIENT_DATA') AS insufficient_data,
        ROUND(
            COUNT(*) FILTER (WHERE outcome = 'HIT_TP')::numeric /
            NULLIF(COUNT(*) FILTER (WHERE outcome IN ('HIT_TP', 'HIT_SL')), 0) * 100,
            1
        ) AS win_rate,
        ROUND(COALESCE(SUM(potential_profit_pips), 0)::numeric, 1) AS net_pips,
        ROUND(AVG(max_favorable_excursion_pips)::numeric, 2) AS avg_mfe,
        ROUND(AVG(max_adverse_excursion_pips)::numeric, 2) AS avg_mae
    FROM smc_rejection_outcomes
    WHERE {where_sql}
    """
    return fetch_one(cur, query, params)


def get_rejection_stage_breakdown(cur, args: argparse.Namespace) -> list[dict[str, Any]]:
    clauses = ["analysis_timestamp >= NOW() - INTERVAL %s"]
    params: list[Any] = [f"{args.days} days"]
    if args.pair:
        clauses.append("pair = %s")
        params.append(args.pair)
    where_sql = " AND ".join(clauses)
    query = f"""
    SELECT
        rejection_stage,
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE outcome = 'HIT_TP') AS winners,
        COUNT(*) FILTER (WHERE outcome = 'HIT_SL') AS losers,
        ROUND(
            COUNT(*) FILTER (WHERE outcome = 'HIT_TP')::numeric /
            NULLIF(COUNT(*) FILTER (WHERE outcome IN ('HIT_TP', 'HIT_SL')), 0) * 100,
            1
        ) AS win_rate,
        ROUND(COALESCE(SUM(potential_profit_pips), 0)::numeric, 1) AS net_pips
    FROM smc_rejection_outcomes
    WHERE {where_sql}
    GROUP BY rejection_stage
    HAVING COUNT(*) >= 3
    ORDER BY net_pips DESC, total DESC
    LIMIT 8
    """
    return fetch_all(cur, query, params)


def get_rejection_pair_breakdown(cur, args: argparse.Namespace) -> list[dict[str, Any]]:
    clauses = ["analysis_timestamp >= NOW() - INTERVAL %s"]
    params: list[Any] = [f"{args.days} days"]
    where_sql = " AND ".join(clauses)
    query = f"""
    SELECT
        pair,
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE outcome = 'HIT_TP') AS winners,
        COUNT(*) FILTER (WHERE outcome = 'HIT_SL') AS losers,
        ROUND(
            COUNT(*) FILTER (WHERE outcome = 'HIT_TP')::numeric /
            NULLIF(COUNT(*) FILTER (WHERE outcome IN ('HIT_TP', 'HIT_SL')), 0) * 100,
            1
        ) AS win_rate,
        ROUND(COALESCE(SUM(potential_profit_pips), 0)::numeric, 1) AS net_pips
    FROM smc_rejection_outcomes
    WHERE {where_sql}
    GROUP BY pair
    HAVING COUNT(*) >= 3
    ORDER BY net_pips DESC, total DESC
    LIMIT 8
    """
    rows = fetch_all(cur, query, params)
    if args.pair:
        return [row for row in rows if row.get("pair") == args.pair]
    return rows


def get_rejected_session_breakdown(cur, args: argparse.Namespace) -> list[dict[str, Any]]:
    clauses = ["analysis_timestamp >= NOW() - INTERVAL %s"]
    params: list[Any] = [f"{args.days} days"]
    if args.pair:
        clauses.append("pair = %s")
        params.append(args.pair)
    where_sql = " AND ".join(clauses)
    query = f"""
    SELECT
        market_session,
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE outcome = 'HIT_TP') AS winners,
        COUNT(*) FILTER (WHERE outcome = 'HIT_SL') AS losers,
        ROUND(
            COUNT(*) FILTER (WHERE outcome = 'HIT_TP')::numeric /
            NULLIF(COUNT(*) FILTER (WHERE outcome IN ('HIT_TP', 'HIT_SL')), 0) * 100,
            1
        ) AS win_rate,
        ROUND(COALESCE(SUM(potential_profit_pips), 0)::numeric, 1) AS net_pips
    FROM smc_rejection_outcomes
    WHERE {where_sql}
      AND market_session IS NOT NULL
    GROUP BY market_session
    HAVING COUNT(*) >= 3
    ORDER BY net_pips DESC, total DESC
    LIMIT 6
    """
    return fetch_all(cur, query, params)


def build_recommendations(
    approved_summary: dict[str, Any],
    rejection_summary: dict[str, Any],
    stage_breakdown: list[dict[str, Any]],
    pair_breakdown: list[dict[str, Any]],
    reason_rows: list[dict[str, Any]],
    chart_summary: dict[str, Any],
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []

    approved_total = int(approved_summary.get("total_closed_trades") or 0)
    approved_win_rate = float(approved_summary.get("win_rate") or 0.0)
    approved_net_pips = float(approved_summary.get("net_pips") or 0.0)

    rejected_total = int(rejection_summary.get("total_analyzed") or 0)
    rejected_win_rate = float(rejection_summary.get("win_rate") or 0.0)
    rejected_net_pips = float(rejection_summary.get("net_pips") or 0.0)

    if rejected_total >= 10 and rejected_win_rate >= 55.0 and rejected_net_pips > 0:
        top_stage = next((row for row in stage_breakdown if float(row.get("net_pips") or 0) > 0), None)
        evidence = (
            f"Rejected cohort has {rejected_win_rate:.1f}% would-be win rate and {rejected_net_pips:+.1f} net pips."
        )
        if top_stage:
            evidence += f" Best positive stage: {top_stage['rejection_stage']} ({float(top_stage.get('net_pips') or 0):+.1f} pips)."
        recommendations.append(
            Recommendation(
                category="Prompt loosening",
                severity="high",
                message="Claude is likely over-rejecting viable setups. Review overly strict language around trend conflict, resistance proximity, or momentum extension.",
                evidence=evidence,
            )
        )

    if approved_total >= 10 and approved_win_rate <= 45.0 and approved_net_pips <= 0:
        recommendations.append(
            Recommendation(
                category="Prompt tightening",
                severity="high",
                message="Approved trades are underperforming. Tighten the approval gate and force stronger path-quality checks before approval.",
                evidence=(
                    f"Approved closed trades show {approved_win_rate:.1f}% win rate and {approved_net_pips:+.1f} net pips."
                ),
            )
        )

    if approved_total >= 10 and rejected_total >= 10 and approved_win_rate < rejected_win_rate:
        recommendations.append(
            Recommendation(
                category="Decision quality",
                severity="high",
                message="Rejected signals are outperforming approved trades. Rework prompt scoring priorities before changing strategy thresholds.",
                evidence=(
                    f"Approved win rate {approved_win_rate:.1f}% vs rejected would-be win rate {rejected_win_rate:.1f}%."
                ),
            )
        )

    positive_pairs = [row for row in pair_breakdown if float(row.get("net_pips") or 0) > 0]
    if positive_pairs:
        lead = positive_pairs[0]
        recommendations.append(
            Recommendation(
                category="Pair-specific tuning",
                severity="medium",
                message=(
                    f"Do not globalize filter changes first. Test pair-specific Claude guidance or thresholds for {lead['pair']}."
                ),
                evidence=(
                    f"{lead['pair']} rejected cohort produced {float(lead.get('net_pips') or 0):+.1f} net pips across {int(lead.get('total') or 0)} outcomes."
                ),
            )
        )

    if reason_rows:
        recommendations.append(
            Recommendation(
                category="Prompt audit",
                severity="medium",
                message="Audit the top recurring rejection reasons against realized outcomes before editing prompt prose.",
                evidence="Top repeated reasons: " + "; ".join(f"{row['reason']} ({row['total']})" for row in reason_rows[:3]),
            )
        )

    chart_total = int(chart_summary.get("total_chart_alerts") or 0)
    chart_fetched = int(chart_summary.get("fetched_chart_alerts") or 0)
    linked_results = int(chart_summary.get("linked_results") or 0)
    if chart_total > 0 and chart_fetched < chart_total:
        recommendations.append(
            Recommendation(
                category="Chart retention",
                severity="medium",
                message="Some alerts reference charts that the analyser could not fetch. Improve chart retention or storage consistency before relying on chart-backed audits.",
                evidence=f"Fetched {chart_fetched}/{chart_total} chart artifacts.",
            )
        )
    elif chart_total > 0 and linked_results < chart_total:
        recommendations.append(
            Recommendation(
                category="Artifact completeness",
                severity="low",
                message="Chart images exist, but some linked prompt/result artifacts are missing. Keep image and prompt/result artifacts together for better forensic review.",
                evidence=f"Linked result artifacts found for {linked_results}/{chart_total} chart-backed alerts.",
            )
        )

    if not recommendations:
        recommendations.append(
            Recommendation(
                category="No strong change",
                severity="low",
                message="The current sample does not justify a prompt change yet. Keep gathering data and review pair- or session-level drift.",
                evidence="No cohort crossed the configured thresholds for a strong recommendation.",
            )
        )

    return recommendations


def render_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.append("No rows available.")
        lines.append("")
        return lines

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(header)
    lines.append(separator)
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def build_report(args: argparse.Namespace, data: dict[str, Any]) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# AI Trade Analyser Report",
        "",
        f"Generated: {now}",
        f"Window: last {args.days} days",
        f"Environment: {args.environment}",
        f"Pair filter: {args.pair or 'All'}",
        f"Strategy filter: {args.strategy or 'All'}",
        "",
        "## Scope",
        "",
        f"- Total alerts scanned: {int(data['alert_stats'].get('total_alerts') or 0)}",
        f"- Claude approved alerts: {int(data['alert_stats'].get('approved_alerts') or 0)}",
        f"- Claude rejected alerts: {int(data['alert_stats'].get('rejected_alerts') or 0)}",
        f"- Average Claude score: {float(data['alert_stats'].get('avg_claude_score') or 0):.2f}",
        "",
        "## Approved Trade Cohort",
        "",
        f"- Closed approved trades: {int(data['approved_summary'].get('total_closed_trades') or 0)}",
        f"- Wins / losses / breakevens: {int(data['approved_summary'].get('wins') or 0)} / {int(data['approved_summary'].get('losses') or 0)} / {int(data['approved_summary'].get('breakevens') or 0)}",
        f"- Win rate: {float(data['approved_summary'].get('win_rate') or 0):.1f}%",
        f"- Net pips: {float(data['approved_summary'].get('net_pips') or 0):+.1f}",
        f"- Average pips per trade: {float(data['approved_summary'].get('avg_pips') or 0):+.2f}",
        f"- Moved to breakeven count: {int(data['approved_summary'].get('moved_to_breakeven_count') or 0)}",
        f"- Partial close count: {int(data['approved_summary'].get('partial_close_count') or 0)}",
        "",
        "## Rejected Outcome Cohort",
        "",
        f"- Analyzed rejected outcomes: {int(data['rejection_summary'].get('total_analyzed') or 0)}",
        f"- Would-be winners / losers: {int(data['rejection_summary'].get('would_be_winners') or 0)} / {int(data['rejection_summary'].get('would_be_losers') or 0)}",
        f"- Would-be win rate: {float(data['rejection_summary'].get('win_rate') or 0):.1f}%",
        f"- Net potential pips: {float(data['rejection_summary'].get('net_pips') or 0):+.1f}",
        f"- Average MFE / MAE: {float(data['rejection_summary'].get('avg_mfe') or 0):.2f} / {float(data['rejection_summary'].get('avg_mae') or 0):.2f}",
        f"- Outcome source: {data['rejection_summary'].get('simulation_source') or 'unknown'}",
        "",
        "## Chart Artifact Coverage",
        "",
        f"- Alerts with stored charts: {int(data['chart_summary'].get('total_chart_alerts') or 0)}",
        f"- Fetchable chart artifacts: {int(data['chart_summary'].get('fetched_chart_alerts') or 0)}",
        f"- Approved / rejected chart-backed alerts: {int(data['chart_summary'].get('approved_chart_alerts') or 0)} / {int(data['chart_summary'].get('rejected_chart_alerts') or 0)}",
        f"- File-backed / URL-backed storage: {int(data['chart_summary'].get('file_backed') or 0)} / {int(data['chart_summary'].get('url_backed') or 0)}",
        f"- Linked prompt/result artifacts: {int(data['chart_summary'].get('linked_prompts') or 0)} / {int(data['chart_summary'].get('linked_results') or 0)}",
        f"- Average chart size: {float(data['chart_summary'].get('avg_chart_bytes') or 0):.1f} bytes",
        f"- Most common dimensions: {data['chart_summary'].get('common_dimensions') or 'Unknown'}",
        "",
        "## Recommendations",
        "",
    ]

    for rec in data["recommendations"]:
        lines.append(
            f"- [{rec.severity.upper()}] {rec.category}: {rec.message} Evidence: {rec.evidence}"
        )
    lines.append("")

    lines.extend(
        render_table(
            "Sample Chart-Backed Alerts",
            data["chart_sample_rows"],
            ["alert_id", "pair", "decision", "storage_type", "fetched", "bytes_size", "dimensions", "linked_result_found", "linked_prompt_found"],
        )
    )
    lines.extend(
        render_table(
            "Top Rejection Reasons",
            data["reason_rows"],
            ["reason", "total"],
        )
    )
    lines.extend(
        render_table(
            "Rejected Outcomes by Stage",
            data["stage_breakdown"],
            ["rejection_stage", "total", "winners", "losers", "win_rate", "net_pips"],
        )
    )
    lines.extend(
        render_table(
            "Rejected Outcomes by Pair",
            data["rejection_pair_breakdown"],
            ["pair", "total", "winners", "losers", "win_rate", "net_pips"],
        )
    )
    lines.extend(
        render_table(
            "Rejected Outcomes by Session",
            data["session_breakdown"],
            ["market_session", "total", "winners", "losers", "win_rate", "net_pips"],
        )
    )
    lines.extend(
        render_table(
            "Approved Trades by Pair",
            data["approved_pair_breakdown"],
            ["pair", "total", "wins", "losses", "win_rate", "net_pips"],
        )
    )

    lines.extend(
        [
            "## Prompt Audit Targets",
            "",
            "- Review `worker/app/forex_scanner/alerts/analysis/prompt_builder.py` against the highest-impact rejection reasons and stage breakdown.",
            "- Compare saved prompt and result artifacts in `worker/app/claude_analysis_enhanced/vision_analysis/` for representative false rejections and false approvals.",
            "- Use the fetched `vision_chart_url` artifacts to confirm the chart-backed subset is complete before changing the prompt based on visual cases.",
            "- Prefer pair-specific or session-specific edits before broad global prompt changes when the positive rejected edge is concentrated.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    with psycopg2.connect(args.dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            pair_risk_config, global_risk_config = load_pair_risk_config(args.config_dsn)
            simulated_rejections = simulate_rejected_alerts(cur, args, pair_risk_config, global_risk_config)
            data = {
                "alert_stats": get_alert_stats(cur, args),
                "approved_summary": get_approved_trade_summary(cur, args),
                "approved_pair_breakdown": get_approved_pair_breakdown(cur, args),
                "reason_rows": get_rejected_alert_reasons(cur, args),
                "rejection_summary": build_simulated_rejection_summary(simulated_rejections),
                "stage_breakdown": [],
                "rejection_pair_breakdown": build_simulated_pair_breakdown(simulated_rejections),
                "session_breakdown": build_simulated_session_breakdown(simulated_rejections),
            }

            chart_rows = get_chart_rows(cur, args)

    chart_summary, chart_artifacts = summarize_chart_artifacts(chart_rows)
    data["chart_summary"] = chart_summary
    data["chart_sample_rows"] = [
        {
            "alert_id": artifact.alert_id,
            "pair": artifact.pair,
            "decision": artifact.decision,
            "storage_type": artifact.storage_type,
            "fetched": artifact.fetched,
            "bytes_size": artifact.bytes_size,
            "dimensions": f"{artifact.width}x{artifact.height}" if artifact.width and artifact.height else "Unknown",
            "linked_result_found": artifact.linked_result_found,
            "linked_prompt_found": artifact.linked_prompt_found,
        }
        for artifact in chart_artifacts[:8]
    ]

    data["recommendations"] = build_recommendations(
        data["approved_summary"],
        data["rejection_summary"],
        data["stage_breakdown"],
        data["rejection_pair_breakdown"],
        data["reason_rows"],
        data["chart_summary"],
    )

    report = build_report(args, data)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(report)
            handle.write("\n")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
