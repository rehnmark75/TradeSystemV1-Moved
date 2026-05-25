"""
Agent Tools - Read-only DB tools for the Claude agent analyzer.

Each function is SELECT-only. Tools are registered as raw JSON schemas
and dispatched via _execute_tool(). All queries target the `forex` and
`strategy_config` databases through the existing psycopg2 connection helper.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_forex_conn():
    """Return a psycopg2 connection to the forex database."""
    import psycopg2
    from forex_scanner.config import DATABASE_URL
    return psycopg2.connect(DATABASE_URL)


def _get_strategy_conn():
    """Return a psycopg2 connection to the strategy_config database."""
    import psycopg2
    from forex_scanner.config import DATABASE_URL
    # Swap database name in the DSN
    url = DATABASE_URL.rsplit("/", 1)[0] + "/strategy_config"
    return psycopg2.connect(url)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def pair_session_wr_recent(
    epic: str,
    hour_bucket: int,
    regime: str,
    lookback_days: int = 30,
    strategy: str = "",
) -> Dict:
    """
    Return WR, PF, avg PnL, and trade count for (epic, hour_bucket, regime, strategy)
    over the last `lookback_days` days, joined from alert_history + trade_log.

    hour_bucket is the UTC hour of the signal (0-23). Pass -1 to aggregate all hours.
    Pass empty string for regime or strategy to aggregate across all values.
    Always pass strategy to avoid mixing results from different strategies on the same pair.
    """
    try:
        conn = _get_forex_conn()
        cur = conn.cursor()

        since = datetime.utcnow() - timedelta(days=lookback_days)

        conditions = ["ah.epic = %s", "ah.alert_timestamp >= %s"]
        params: List[Any] = [epic, since]

        if strategy:
            conditions.append("ah.strategy ILIKE %s")
            params.append(strategy)

        if hour_bucket >= 0:
            conditions.append("EXTRACT(HOUR FROM ah.alert_timestamp)::int = %s")
            params.append(hour_bucket)

        if regime:
            conditions.append("ah.market_regime ILIKE %s")
            params.append(regime)

        where = " AND ".join(conditions)

        cur.execute(
            f"""
            SELECT
                COUNT(*)                                            AS total,
                SUM(CASE WHEN tl.profit_loss > 0 THEN 1 ELSE 0 END) AS wins,
                ROUND(AVG(tl.profit_loss)::numeric, 2)              AS avg_pnl,
                ROUND(
                    CASE
                        WHEN SUM(CASE WHEN tl.profit_loss < 0 THEN ABS(tl.profit_loss) ELSE 0 END) = 0
                        THEN NULL
                        ELSE SUM(CASE WHEN tl.profit_loss > 0 THEN tl.profit_loss ELSE 0 END) /
                             SUM(CASE WHEN tl.profit_loss < 0 THEN ABS(tl.profit_loss) ELSE 0 END)
                    END::numeric, 2
                )                                                   AS profit_factor
            FROM alert_history ah
            JOIN trade_log tl ON tl.alert_id = ah.id
            WHERE {where}
            """,
            params,
        )

        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return {"epic": epic, "trade_count": 0, "win_rate_pct": None, "profit_factor": None, "avg_pnl_gbp": None}

        total = row[0] or 0
        wins = row[1] or 0
        avg_pnl = float(row[2]) if row[2] is not None else None
        pf = float(row[3]) if row[3] is not None else None
        wr = round(wins / total * 100, 1) if total > 0 else None

        return {
            "epic": epic,
            "strategy": strategy or "all",
            "hour_bucket": hour_bucket if hour_bucket >= 0 else "all",
            "regime": regime or "all",
            "lookback_days": lookback_days,
            "trade_count": total,
            "win_rate_pct": wr,
            "profit_factor": pf,
            "avg_pnl_gbp": avg_pnl,
        }

    except Exception as e:
        logger.exception("pair_session_wr_recent failed")
        return {"error": str(e)}


def rejection_density(epic: str, lookback_hours: int = 4) -> Dict:
    """
    Return rejection count + distinct rejection-reason count from
    smc_simple_rejections for `epic` over the last `lookback_hours`.

    High rejection density (>30 rejections, many distinct reasons) indicates
    structure churn — a warning sign for new entries.
    """
    try:
        conn = _get_forex_conn()
        cur = conn.cursor()

        since = datetime.utcnow() - timedelta(hours=lookback_hours)

        cur.execute(
            """
            SELECT
                COUNT(*)                          AS total_rejections,
                COUNT(DISTINCT rejection_reason)  AS distinct_reasons,
                COUNT(DISTINCT attempted_direction) AS signal_types
            FROM smc_simple_rejections
            WHERE epic = %s AND scan_timestamp >= %s
            """,
            (epic, since),
        )
        row = cur.fetchone()

        # Fetch top 5 reasons for context
        cur.execute(
            """
            SELECT rejection_reason, COUNT(*) AS cnt
            FROM smc_simple_rejections
            WHERE epic = %s AND scan_timestamp >= %s
            GROUP BY rejection_reason
            ORDER BY cnt DESC
            LIMIT 5
            """,
            (epic, since),
        )
        top_reasons = [{"reason": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.close()
        conn.close()

        total_rej = int(row[0] or 0) if row else 0
        distinct = int(row[1] or 0) if row else 0
        sig_types = int(row[2] or 0) if row else 0

        return {
            "epic": epic,
            "lookback_hours": lookback_hours,
            "total_rejections": total_rej,
            "distinct_reasons": distinct,
            "signal_types": sig_types,
            "top_reasons": top_reasons,
            "churn_warning": total_rej > 30 and distinct > 10,
        }

    except Exception as e:
        logger.exception("rejection_density failed")
        return {"error": str(e)}


def nearby_lpf_blocks(
    epic: str,
    signal_type: str,
    hour: int,
    regime: str,
    lookback_days: int = 14,
) -> Dict:
    """
    Return recent LPF block decisions for similar setups (same epic,
    signal_type, ±2 hour, same regime) from loss_prevention_decisions.

    Useful even when the current signal passes LPF — a nearby cluster of
    blocks signals a hostile environment.
    """
    try:
        conn = _get_strategy_conn()
        cur = conn.cursor()

        since = datetime.utcnow() - timedelta(days=lookback_days)

        cur.execute(
            """
            SELECT
                COUNT(*)                                         AS total,
                SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END) AS blocks,
                ROUND(AVG(total_penalty)::numeric, 3)            AS avg_penalty,
                MAX(total_penalty)                               AS max_penalty
            FROM loss_prevention_decisions
            WHERE epic = %s
              AND signal_type = %s
              AND ABS(EXTRACT(HOUR FROM signal_timestamp)::int - %s) <= 2
              AND signal_timestamp >= %s
            """,
            (epic, signal_type.upper(), hour, since),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        total = int(row[0] or 0) if row else 0
        blocks = int(row[1] or 0) if row else 0
        block_rate = round(blocks / total * 100, 1) if total > 0 else None
        avg_penalty = float(row[2]) if row and row[2] is not None else None
        max_penalty = float(row[3]) if row and row[3] is not None else None

        return {
            "epic": epic,
            "signal_type": signal_type,
            "hour": hour,
            "regime": regime,
            "lookback_days": lookback_days,
            "similar_signals": total,
            "lpf_blocks": blocks,
            "block_rate_pct": block_rate,
            "avg_penalty": avg_penalty,
            "max_penalty": max_penalty,
        }

    except Exception as e:
        logger.exception("nearby_lpf_blocks failed")
        return {"error": str(e)}


def prior_pair_pnl_today(epic: str) -> Dict:
    """
    Return today's (UTC date) executed trade PnL for `epic`.

    Captures intra-day autocorrelation — pairs that are bleeding today
    often continue bleeding (news, spread, regime misclassification).
    """
    try:
        conn = _get_forex_conn()
        cur = conn.cursor()

        today = datetime.utcnow().date()

        cur.execute(
            """
            SELECT
                COUNT(*)                                             AS trades,
                SUM(profit_loss)                                     AS total_pnl,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)    AS wins,
                MIN(profit_loss)                                     AS worst_loss,
                MAX(profit_loss)                                     AS best_win
            FROM trade_log
            WHERE symbol = %s
              AND DATE(timestamp AT TIME ZONE 'UTC') = %s
              AND status = 'closed'
            """,
            (epic, today),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        trades = int(row[0] or 0) if row else 0
        total_pnl = float(row[1]) if row and row[1] is not None else 0.0
        wins = int(row[2] or 0) if row else 0
        wr = round(wins / trades * 100, 1) if trades > 0 else None

        return {
            "epic": epic,
            "date_utc": str(today),
            "trades_today": trades,
            "total_pnl_gbp": round(total_pnl, 2),
            "win_rate_pct": wr,
            "worst_loss_gbp": float(row[3]) if row and row[3] is not None else None,
            "best_win_gbp": float(row[4]) if row and row[4] is not None else None,
        }

    except Exception as e:
        logger.exception("prior_pair_pnl_today failed")
        return {"error": str(e)}


# Allowlist: strategy name → its pair_overrides table. Used by get_pair_config.
_STRATEGY_CONFIG_TABLE: Dict[str, str] = {
    "SMC_SIMPLE":       "smc_simple_pair_overrides",
    "XAU_GOLD":         "xau_gold_pair_overrides",
    "IMPULSE_FADE":     "impulse_fade_pair_overrides",
    "DONCHIAN_TURTLE":  "donchian_turtle_pair_overrides",
    "MEAN_REVERSION":   "mean_reversion_pair_overrides",
    "SMC_MOMENTUM":     "smc_momentum_pair_overrides",
    "RANGE_FADE":       "range_fade_pair_overrides",
    "FA_OR_ATR_TRAIL":  "fa_or_atr_trail_pair_overrides",
}

# Strategies whose pair_overrides table has a config_set column (demo/live split).
# Tables without this column have one row per epic and don't need the filter.
_STRATEGY_HAS_CONFIG_SET: frozenset = frozenset({
    "XAU_GOLD", "MEAN_REVERSION", "SMC_MOMENTUM",
    "RANGE_FADE", "FA_OR_ATR_TRAIL",
})

# Per-strategy SQL fragment for the four "extended" columns that not all tables have.
# Table names come from the allowlist above so there is no injection risk.
_STRATEGY_EXTENDED_COLS: Dict[str, str] = {
    # Most strategies have all four columns (SMC_SIMPLE, XAU_GOLD, IMPULSE_FADE,
    # DONCHIAN_TURTLE, MEAN_REVERSION):
    "__default__": "fixed_stop_loss_pips, fixed_take_profit_pips, min_confidence, max_confidence",
    # RANGE_FADE has SL/TP but no confidence columns:
    "RANGE_FADE":      "fixed_stop_loss_pips, fixed_take_profit_pips, NULL AS min_confidence, NULL AS max_confidence",
    # SMC_MOMENTUM has min_confidence only — no SL/TP or max_confidence:
    "SMC_MOMENTUM":    "NULL AS fixed_stop_loss_pips, NULL AS fixed_take_profit_pips, min_confidence, NULL AS max_confidence",
    # FA_OR_ATR_TRAIL uses ATR-based stops — none of the four standard columns exist:
    "FA_OR_ATR_TRAIL": "NULL AS fixed_stop_loss_pips, NULL AS fixed_take_profit_pips, NULL AS min_confidence, NULL AS max_confidence",
}


def get_pair_config(epic: str, strategy: str = "SMC_SIMPLE") -> Dict:
    """
    Return the pair configuration row for `epic` from the correct strategy table.

    Always pass `strategy` so the correct table is queried — each strategy has its
    own pair_overrides table (smc_simple_pair_overrides, xau_gold_pair_overrides,
    impulse_fade_pair_overrides, donchian_turtle_pair_overrides,
    mean_reversion_pair_overrides, smc_momentum_pair_overrides,
    range_fade_pair_overrides, fa_or_atr_trail_pair_overrides).
    Returns monitor_only flag, SL/TP pips, and confidence thresholds.
    """
    strategy_upper = strategy.upper()
    table = _STRATEGY_CONFIG_TABLE.get(strategy_upper)
    if not table:
        return {"error": f"Unknown strategy '{strategy}'. Known: {list(_STRATEGY_CONFIG_TABLE)}", "epic": epic}

    try:
        conn = _get_strategy_conn()
        cur = conn.cursor()

        # smc_simple_pair_overrides stores monitor_only inside the JSONB parameter_overrides
        # column rather than as a direct boolean column. All other strategy tables use a
        # direct boolean monitor_only column.
        if table == "smc_simple_pair_overrides":
            monitor_col = "(parameter_overrides->>'monitor_only')::boolean"
        else:
            monitor_col = "monitor_only"

        extended_cols = _STRATEGY_EXTENDED_COLS.get(strategy_upper, _STRATEGY_EXTENDED_COLS["__default__"])

        # Tables with a config_set column have separate demo/live rows — always
        # query demo so the agent sees the active trading config, not the live row
        # (which is always disabled until manual promotion).
        config_set_clause = "AND config_set = 'demo'" if strategy_upper in _STRATEGY_HAS_CONFIG_SET else ""

        # Table name and extended_cols come from allowlists — no injection risk.
        cur.execute(
            f"""
            SELECT
                epic,
                is_enabled,
                {monitor_col}           AS monitor_only,
                {extended_cols},
                parameter_overrides
            FROM {table}
            WHERE epic = %s {config_set_clause}
            ORDER BY id DESC
            LIMIT 1
            """,
            (epic,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return {"epic": epic, "strategy": strategy, "found": False, "table": table}

        overrides = row[7]
        if isinstance(overrides, str):
            try:
                overrides = json.loads(overrides)
            except Exception:
                overrides = {}

        return {
            "epic": row[0],
            "strategy": strategy,
            "table": table,
            "found": True,
            "is_enabled": row[1],
            "monitor_only": bool(row[2]) if row[2] is not None else False,
            "fixed_stop_loss_pips": row[3],
            "fixed_take_profit_pips": row[4],
            "min_confidence": float(row[5]) if row[5] else None,
            "max_confidence": float(row[6]) if row[6] else None,
            "parameter_overrides": overrides or {},
        }

    except Exception as e:
        logger.exception("get_pair_config failed")
        return {"error": str(e)}


def render_chart(epic: str, timeframes: Optional[List[str]] = None) -> Dict:
    """
    Generate a multi-timeframe chart for `epic` and return it as base64 PNG.

    Only call this when visual chart analysis is genuinely needed — it costs
    tokens and latency. Defaults to ["4h", "1h", "15m"].
    """
    if timeframes is None:
        timeframes = ["4h", "1h", "15m"]

    try:
        import psycopg2
        import pandas as pd
        from forex_scanner.config import DATABASE_URL
        from forex_scanner.alerts.forex_chart_generator import ForexChartGenerator

        candle_dict: Dict[str, Any] = {}
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        tf_minutes = {"4h": 240, "1h": 60, "15m": 15, "5m": 5}
        for tf in timeframes:
            mins = tf_minutes.get(tf)
            if not mins:
                continue
            cur.execute(
                """
                SELECT start_time, open_price, high_price, low_price, close_price, volume
                FROM ig_candles
                WHERE epic = %s AND timeframe = %s
                ORDER BY start_time DESC LIMIT 100
                """,
                (epic, tf if tf != "4h" else "4h"),
            )
            rows = cur.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["start_time", "open", "high", "low", "close", "volume"])
                df["start_time"] = pd.to_datetime(df["start_time"])
                df = df.sort_values("start_time").reset_index(drop=True)
                candle_dict[tf] = df

        cur.close()
        conn.close()

        if not candle_dict:
            return {"error": "No candle data found in ig_candles for chart generation"}

        chart_gen = ForexChartGenerator()
        chart_b64 = chart_gen.generate_signal_chart(
            epic=epic,
            candles=candle_dict,
            signal={"epic": epic},
        )

        if not chart_b64:
            return {"error": "Chart generator returned empty result"}

        return {
            "epic": epic,
            "timeframes": list(candle_dict.keys()),
            "chart_base64": chart_b64,
            "size_bytes": len(chart_b64),
        }

    except Exception as e:
        logger.exception("render_chart failed")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool registry (JSON schema definitions for the Anthropic API)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "pair_session_wr_recent",
        "description": (
            "Look up historical win rate, profit factor, and average PnL for a specific "
            "pair × strategy × UTC hour × market regime over a recent lookback window. "
            "Always pass the strategy name so results are scoped to this strategy only — "
            "IMPULSE_FADE losses on EURJPY at hour 20 must not contaminate SMC_SIMPLE WR at hour 8. "
            "Use this first to detect adverse environments before approving a signal. "
            "Key warning: XAU_GOLD in 'ranging' regime has historically shown <25% WR. "
            "Supported strategy values: SMC_SIMPLE, XAU_GOLD, IMPULSE_FADE, DONCHIAN_TURTLE, "
            "MEAN_REVERSION, SMC_MOMENTUM, RANGE_FADE, FA_OR_ATR_TRAIL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epic": {"type": "string", "description": "Full IG epic, e.g. CS.D.EURUSD.CEEM.IP"},
                "hour_bucket": {
                    "type": "integer",
                    "description": "UTC hour 0-23 of the signal. Pass -1 to aggregate all hours.",
                },
                "regime": {
                    "type": "string",
                    "description": "Market regime label, e.g. 'trending', 'ranging', 'expansion'. Empty string for all.",
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Days of history to include. Default 30.",
                    "default": 30,
                },
                "strategy": {
                    "type": "string",
                    "description": (
                        "Strategy name to filter by, e.g. 'SMC_SIMPLE', 'XAU_GOLD', 'IMPULSE_FADE', "
                        "'DONCHIAN_TURTLE', 'MEAN_REVERSION', 'SMC_MOMENTUM', 'RANGE_FADE', "
                        "'FA_OR_ATR_TRAIL'. Empty string aggregates all strategies (not recommended)."
                    ),
                    "default": "",
                },
            },
            "required": ["epic", "hour_bucket", "regime"],
        },
    },
    {
        "name": "rejection_density",
        "description": (
            "Count recent SMC structure rejections for a pair over the last N hours. "
            "NOTE: this table is populated by SMC_SIMPLE only — do not use for XAU_GOLD, "
            "IMPULSE_FADE, DONCHIAN_TURTLE, or MEAN_REVERSION signals. "
            "High rejection density (>30 rejections, many distinct reasons) indicates "
            "choppy structure — a warning sign for new entries on EURUSD and JPY pairs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epic": {"type": "string", "description": "Full IG epic"},
                "lookback_hours": {
                    "type": "integer",
                    "description": "Hours to look back. Default 4.",
                    "default": 4,
                },
            },
            "required": ["epic"],
        },
    },
    {
        "name": "nearby_lpf_blocks",
        "description": (
            "Check how often the Loss Prevention Filter has blocked similar setups "
            "(same pair, signal type, nearby hour, same regime) recently. "
            "A high block rate on similar signals is a red flag even if this individual "
            "signal passed the LPF."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epic": {"type": "string"},
                "signal_type": {"type": "string", "description": "BUY or SELL"},
                "hour": {"type": "integer", "description": "UTC hour of signal"},
                "regime": {"type": "string", "description": "Market regime label"},
                "lookback_days": {
                    "type": "integer",
                    "description": "Days of history. Default 14.",
                    "default": 14,
                },
            },
            "required": ["epic", "signal_type", "hour", "regime"],
        },
    },
    {
        "name": "prior_pair_pnl_today",
        "description": (
            "Return today's executed trade PnL for this pair (UTC date). "
            "Useful to detect intra-day bleeding — a pair that is significantly "
            "negative today often indicates adverse conditions that persist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epic": {"type": "string", "description": "Full IG epic"},
            },
            "required": ["epic"],
        },
    },
    {
        "name": "get_pair_config",
        "description": (
            "Retrieve the pair configuration for a specific strategy: monitor_only flag, "
            "SL/TP pips, and confidence thresholds. Each strategy has its own config table — "
            "always pass the strategy name so the correct table is queried. "
            "Supported strategies: SMC_SIMPLE, XAU_GOLD, IMPULSE_FADE, DONCHIAN_TURTLE, "
            "MEAN_REVERSION, SMC_MOMENTUM, RANGE_FADE, FA_OR_ATR_TRAIL. "
            "Note: FA_OR_ATR_TRAIL uses ATR-based stops so fixed_stop_loss_pips will be null. "
            "Call this to verify the pair is actively traded before approving."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epic": {"type": "string", "description": "Full IG epic"},
                "strategy": {
                    "type": "string",
                    "description": (
                        "Strategy name, e.g. 'SMC_SIMPLE', 'XAU_GOLD', 'IMPULSE_FADE', "
                        "'DONCHIAN_TURTLE', 'MEAN_REVERSION', 'SMC_MOMENTUM', "
                        "'RANGE_FADE', 'FA_OR_ATR_TRAIL'. Required."
                    ),
                },
            },
            "required": ["epic", "strategy"],
        },
    },
    {
        "name": "render_chart",
        "description": (
            "Generate a multi-timeframe price chart for a pair and return it as a base64 PNG. "
            "Only call this when you need visual confirmation of chart structure. "
            "It costs additional tokens — call it last, not first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epic": {"type": "string", "description": "Full IG epic"},
                "timeframes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of timeframes, e.g. ['4h', '1h', '15m']. Defaults to ['4h', '1h', '15m'].",
                },
            },
            "required": ["epic"],
        },
    },
]


# Dispatch map: tool name → callable
TOOL_DISPATCH = {
    "pair_session_wr_recent": pair_session_wr_recent,
    "rejection_density": rejection_density,
    "nearby_lpf_blocks": nearby_lpf_blocks,
    "prior_pair_pnl_today": prior_pair_pnl_today,
    "get_pair_config": get_pair_config,
    "render_chart": render_chart,
}


def execute_tool(name: str, inputs: Dict) -> Any:
    """Dispatch a tool call by name, returning a JSON-serialisable result."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**inputs)
    except TypeError as e:
        return {"error": f"Bad tool arguments for {name}: {e}"}
