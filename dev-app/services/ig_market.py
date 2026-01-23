import httpx
from config import API_BASE_URL
from sqlalchemy.orm import Session
from .models import Candle
from datetime import timedelta


async def get_current_bid_price(auth_headers, epic):
    headers = {
        "Accept": "application/json; charset=UTF-8",
        "Content-Type": "application/json; charset=UTF-8",
        "X-IG-API-KEY": auth_headers["X-IG-API-KEY"],
        "CST": auth_headers["CST"],
        "X-SECURITY-TOKEN": auth_headers["X-SECURITY-TOKEN"],
        "Version": "3"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/markets/{epic}", headers=headers)
        response.raise_for_status()
        data = response.json()

        bid_price = data["snapshot"]["bid"]
        offer_price = data["snapshot"]["offer"]  # Ask/offer price for BUY trades
        currency_symbol = data["instrument"]["currencies"][0].get("symbol", "")
        currency_code = data["instrument"]["currencies"][0].get("code", "")
        min_distance = data["dealingRules"].get("minNormalStopOrLimitDistance", {}).get("value", None)


        try:
            one_pip = float(data["instrument"]["onePipMeans"].split()[0])
        except (KeyError, ValueError, IndexError):
            one_pip = 0.0001

        # Calculate spread in price units and pips
        spread_price = offer_price - bid_price
        spread_pips = spread_price / one_pip

        return {
            "bid_price": bid_price,
            "offer_price": offer_price,
            "spread_price": spread_price,
            "spread_pips": spread_pips,
            "currency": currency_symbol,
            "point_size": one_pip,
            "currency_code": currency_code,
            "min_distance": min_distance
        }


# Get candle size of the last 15 min candle in IG points
async def get_last_15m_candle_range(auth_headers, epic):
    url = f"{API_BASE_URL}/prices/{epic}?resolution=MINUTE_15"
    headers = {
        "Accept": "application/json; charset=UTF-8",
        "Content-Type": "application/json; charset=UTF-8",
        "X-IG-API-KEY": auth_headers["X-IG-API-KEY"],
        "CST": auth_headers["CST"],
        "X-SECURITY-TOKEN": auth_headers["X-SECURITY-TOKEN"],
        "Version": "3"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "prices" not in data or not data["prices"]:
            raise ValueError("No candle data available.")

        candle = data["prices"][0]
        high = candle["highPrice"]["ask"]
        low = candle["lowPrice"]["bid"]
        range_price = high - low

        # Convert to IG points based on decimalPlacesFactor
        snapshot = data.get("snapshot") or {}
        scaling_factor = snapshot.get("scalingFactor")

        if not scaling_factor:
            market_url = f"{API_BASE_URL}/markets/{epic}"
            market_response = await client.get(market_url, headers=headers)
            market_response.raise_for_status()
            market_data = market_response.json()
            scaling_factor = market_data.get("snapshot", {}).get("scalingFactor", 100)

        return round(range_price * scaling_factor)

async def get_last_15m_candle_range_local(db: Session, epic: str) -> int:
    candles = (
        db.query(Candle)
        .filter(Candle.epic == epic, Candle.timeframe == 5)
        .order_by(Candle.start_time.desc())
        .limit(3)
        .all()
    )

    if len(candles) < 3:
        print(f"⚠️ Only {len(candles)} 5m candles found for {epic}, returning fallback range = 10")
        return 10  # Fallback value if not enough data

    candles = list(reversed(candles))
    high = max(c.high for c in candles)
    low = min(c.low for c in candles)
    range_price = high - low

    # 🧠 Dynamic scaling based on price magnitude
    if high > 100:
        scaling_factor = 100  # for JPY and high-value instruments
    else:
        scaling_factor = 10000  # for most FX pairs

    return round(range_price * scaling_factor)

