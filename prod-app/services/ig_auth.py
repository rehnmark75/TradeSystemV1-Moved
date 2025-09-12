# services/ig_auth.py
import time
import json
import httpx
from services.keyvault import get_secret
from config import API_BASE_URL

_token_cache = {
    "CST": None,
    "X-SECURITY-TOKEN": None,
    "expires_at": 0
}


async def ig_login(api_key: str, ig_pwd: str, ig_usr: str, api_url: str = API_BASE_URL, cache_ttl: int = 3600):
    if _token_cache["CST"] and time.time() < _token_cache["expires_at"]:
        print("Reusing cached IG token")
        return {
            "CST": _token_cache["CST"],
            "X-SECURITY-TOKEN": _token_cache["X-SECURITY-TOKEN"]
        }

    headers = {
        "Accept": "application/json; charset=UTF-8",
        "Content-Type": "application/json; charset=UTF-8",
        "X-IG-API-KEY": api_key,
        "Version": "2",
        "User-Agent": "IG Python Client"
    }

    payload = {
        "identifier": ig_usr,
        "password": ig_pwd,
        "encryptedPassword": False
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_url}/session",
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()

        cst = response.headers.get("CST")
        xst = response.headers.get("X-SECURITY-TOKEN")

        print("✅ IG Login successful")

        _token_cache["CST"] = cst
        _token_cache["X-SECURITY-TOKEN"] = xst
        _token_cache["expires_at"] = time.time() + cache_ttl

        return {
            "CST": cst,
            "X-SECURITY-TOKEN": xst
        }
