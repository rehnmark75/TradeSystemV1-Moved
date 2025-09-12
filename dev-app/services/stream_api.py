import httpx

from config import FASTAPI_STREAM_URL
STREAM_API_URL = FASTAPI_STREAM_URL  # 👈 service name in docker-compose

async def trigger_trade_tracking(
    deal_id: str,
    epic: str,
    entry_price: float,
    direction: str,
    headers: dict,
    trailing_distance: int = 15
):
    url = f"{STREAM_API_URL}/stream/track-trade/{deal_id}"
    payload = {
        "epic": epic,
        "entry_price": entry_price,
        "direction": direction,
        "headers": headers,
        "trailing_distance": trailing_distance
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
