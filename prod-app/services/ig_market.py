import httpx
from config import API_BASE_URL

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
        currency_symbol = data["instrument"]["currencies"][0].get("symbol", "")
        currency_code = data["instrument"]["currencies"][0].get("code", "")

        try:
            one_pip = float(data["instrument"]["onePipMeans"].split()[0])
        except (KeyError, ValueError, IndexError):
            one_pip = 0.0001

        return {
            "bid_price": bid_price,
            "currency": currency_symbol,
            "point_size": one_pip,
            "currency_code": currency_code
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