import httpx
import json
from config import API_BASE_URL

async def has_open_position(epic: str, auth_headers: dict) -> bool:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/positions", headers=auth_headers)
        response.raise_for_status()
        positions = response.json().get("positions", [])
        return any(pos["market"]["epic"] == epic for pos in positions)


async def place_market_order(auth_headers, market_epic, direction, currency_code):
    payload = {
        "epic": market_epic,
        "expiry": "-",
        "direction": direction,
        "size": 1.0,
        "orderType": "MARKET",
        "guaranteedStop": False,
        "forceOpen": True,
        "currencyCode": currency_code,
        "stopDistance": 15,
        "limitDistance": 15
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE_URL}/positions/otc",
            headers=auth_headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        return response.json()