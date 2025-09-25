# services/data.py

from sqlalchemy import text
import pandas as pd
import numpy as np

def get_candle_data(engine, tf, epic, limit=500):
    limit = int(limit)
    sql = f"""
        SELECT * FROM (
            SELECT start_time, open, high, low, close
            FROM ig_candles
            WHERE timeframe = :tf AND epic = :epic
                AND NOT (open = high AND high = low AND low = close)
            ORDER BY start_time DESC
            LIMIT {limit}
        ) sub
        ORDER BY start_time ASC
    """
    query = text(sql)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tf": tf, "epic": epic})
    return df



def get_trade_logs(engine, epic, min_time):
    # Convert min_time to a naive datetime if it has timezone info
    if hasattr(min_time, 'tz_localize'):
        min_time = min_time.tz_convert('UTC').tz_localize(None)

    query = text("""
        SELECT
            tl.timestamp,
            tl.entry_price,
            tl.direction,
            tl.profit_loss,
            tl.status,
            tl.sl_price,
            tl.tp_price,
            tl.moved_to_breakeven,
            tl.trigger_distance,
            tl.last_trigger_price,
            ah.strategy,
            ah.signal_type,
            ah.confidence_score,
            ah.strategy_metadata,
            (ah.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'dominant_regime') as regime,
            (ah.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'confidence')::float as regime_confidence,
            (ah.strategy_metadata::json->'market_intelligence'->'session_analysis'->>'current_session') as session,
            (ah.strategy_metadata::json->'market_intelligence'->>'volatility_level') as volatility_level,
            (ah.strategy_metadata::json->'market_intelligence'->>'intelligence_source') as intelligence_source
        FROM trade_log tl
        LEFT JOIN alert_history ah ON tl.alert_id = ah.id
        WHERE tl.symbol = :epic AND tl.timestamp >= :min_time
        ORDER BY tl.timestamp
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"epic": epic, "min_time": min_time})
    return df

def get_epics(engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT epic FROM ig_candles ORDER BY epic"))
        return [row[0] for row in result.fetchall()]

def robust_clean_candle_data(df: pd.DataFrame, tz) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    numeric_cols = ["open", "high", "low", "close"]

    # Convert to numeric, coerce errors
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaNs
    df.dropna(subset=numeric_cols, inplace=True)

    # Enforce float64
    df[numeric_cols] = df[numeric_cols].astype("float64")

    # Remove flat candles
    is_flat = (
        np.isclose(df["open"], df["high"]) &
        np.isclose(df["high"], df["low"]) &
        np.isclose(df["low"], df["close"])
    )
    df = df[~is_flat]

    df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize("UTC").dt.tz_convert(tz)
    df.sort_values("start_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
