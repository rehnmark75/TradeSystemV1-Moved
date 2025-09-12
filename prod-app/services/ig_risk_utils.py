import httpx
import pandas as pd
from config import API_BASE_URL

async def get_atr(epic: str, trading_headers: dict, resolution: str = "MINUTE_15", periods: int = 14) -> float:
    """
    Fetch historical price data from IG and calculate the ATR (Average True Range).
    """
    limit = periods + 1
    url = f"{API_BASE_URL}/deal/prices/{epic}?resolution={resolution}&max={limit}&pageSize={limit}"

    headers = {
        **trading_headers,
        "Version": "3"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        prices = response.json()["prices"]

    df = pd.DataFrame([{
        "high": p["highPrice"]["bid"],
        "low": p["lowPrice"]["bid"],
        "close": p["closePrice"]["bid"]
    } for p in prices])

    if len(df) < periods:
        raise ValueError(f"Not enough data to calculate ATR({periods})")

    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df[["high", "prev_close"]].max(axis=1) - df[["low", "prev_close"]].min(axis=1)
    atr = df["tr"].rolling(window=periods).mean().iloc[-1]
    return round(float(atr), 5)


async def calculate_dynamic_sl_tp(epic: str, trading_headers: dict, atr: float, rr_ratio: float = 2.0) -> dict:
    """
    Calculate valid stop-loss and take-profit distances based on ATR and IG market constraints.
    """
    url = f"https://api.ig.com/gateway/deal/markets/{epic}"

    headers = {
        **trading_headers,
        "Version": "3"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        response_json = response.json()

    instrument = response_json["instrument"]
    dealing_rules = response_json["dealingRules"]
    print("Dealing rules keys:", dealing_rules.keys())
    min_stop = float(dealing_rules["minNormalStopOrLimitDistance"]["value"])
    stop_step = float(dealing_rules["minStepDistance"]["value"])

    raw_stop = atr * 1.5
    valid_stop = max(min_stop, round(raw_stop / stop_step) * stop_step)

    raw_limit = valid_stop * rr_ratio
    valid_limit = round(raw_limit / stop_step) * stop_step

    return {
        "stopDistance": str(valid_stop),
        "limitDistance": str(valid_limit)
    }
