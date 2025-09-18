# utils/helpers.py
import pandas as pd
import numpy as np
from services.data import get_candle_data


def clean_candle_data(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["open", "high", "low", "close"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=numeric_cols, inplace=True)
    df[numeric_cols] = df[numeric_cols].astype("float64")

    is_flat = (
        np.isclose(df["open"].values, df["high"].values) &
        np.isclose(df["high"].values, df["low"].values) &
        np.isclose(df["low"].values, df["close"].values)
    )
    df = df[~is_flat]

    df["start_time"] = pd.to_datetime(df["start_time"]).dt.tz_localize("UTC")
    return df


def synthesize_15m_from_5m(engine, epic: str, lookback_candles: int) -> pd.DataFrame:
    df_5m = get_candle_data(engine, 5, epic, limit=lookback_candles * 3)
    df_5m = clean_candle_data(df_5m)
    df_5m.set_index("start_time", inplace=True)

    df_15m = df_5m.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()

    return df_15m


def synthesize_60m_from_5m(engine, epic: str, lookback_candles: int) -> pd.DataFrame:
    """
    Synthesize 60-minute candles from 5-minute candle data

    Args:
        engine: Database engine
        epic: Trading pair epic code
        lookback_candles: Number of 60m candles to synthesize

    Returns:
        DataFrame with synthesized 60-minute candles
    """
    # Need 12x more 5m candles to create the requested 60m candles
    df_5m = get_candle_data(engine, 5, epic, limit=lookback_candles * 12)
    df_5m = clean_candle_data(df_5m)
    df_5m.set_index("start_time", inplace=True)

    df_60m = df_5m.resample("60min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()

    return df_60m


def compute_trend_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame):
    trend_1h = trend_4h = overall_trend = None

    if not df_4h.empty:
        df_4h["ema_50"] = df_4h["close"].ewm(span=50, adjust=False).mean()
        trend_4h = "up" if df_4h["close"].iloc[-1] > df_4h["ema_50"].iloc[-1] else "down"

    if not df_1h.empty:
        df_1h["ema_50"] = df_1h["close"].ewm(span=50, adjust=False).mean()
        trend_1h = "up" if df_1h["close"].iloc[-1] > df_1h["ema_50"].iloc[-1] else "down"
        overall_trend = "uptrend" if trend_1h == "up" else "downtrend"

    return trend_1h, trend_4h, overall_trend


def filter_by_time(data, start, end, time_key="time"):
    return [d for d in data if start <= d[time_key] <= end]

def filter_df_by_time(df, start, end, time_col="timestamp"):
    return df[(df[time_col] >= start) & (df[time_col] <= end)].copy()