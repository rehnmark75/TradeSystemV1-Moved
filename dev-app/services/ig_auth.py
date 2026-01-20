# services/ig_auth.py
import time
import json
import asyncio
import httpx
# Note: get_secret is imported by callers but not directly used here
from config import API_BASE_URL

# Separate caches for demo vs production to prevent token collision
# Key: API URL, Value: token cache dict
_token_caches = {}

# Login timeout and retry settings
LOGIN_TIMEOUT_SECONDS = 30.0
LOGIN_MAX_RETRIES = 3
LOGIN_MAX_RETRIES_STARTUP = 1  # Reduced retries during startup to prevent blocking
LOGIN_RETRY_BACKOFF = [2, 5, 10]  # Seconds to wait between retries


def _get_cache_for_url(api_url: str) -> dict:
    """Get or create token cache for specific API URL."""
    if api_url not in _token_caches:
        _token_caches[api_url] = {
            "CST": None,
            "X-SECURITY-TOKEN": None,
            "ACCOUNT_ID": None,
            "STREAMING_URL": None,
            "expires_at": 0
        }
    return _token_caches[api_url]


async def ig_login(api_key: str, ig_pwd: str, ig_usr: str, api_url: str = API_BASE_URL, cache_ttl: int = 3600, startup_mode: bool = False):
    """
    Login to IG API with timeout and retry logic.

    Features:
    - 30 second timeout to prevent indefinite hangs
    - 3 retries with exponential backoff (2s, 5s, 10s) in normal mode
    - 1 retry in startup_mode to prevent blocking container initialization
    - URL-specific caching to avoid demo/production token collision

    Args:
        startup_mode: If True, use reduced retries to prevent blocking during container startup
    """
    # Use URL-specific cache to avoid demo/production token collision
    cache = _get_cache_for_url(api_url)

    if cache["CST"] and time.time() < cache["expires_at"]:
        env_type = "PRODUCTION" if "api.ig.com" in api_url else "DEMO"
        print(f"Reusing cached IG token ({env_type})")
        return {
            "CST": cache["CST"],
            "X-SECURITY-TOKEN": cache["X-SECURITY-TOKEN"],
            "ACCOUNT_ID": cache.get("ACCOUNT_ID"),
            "STREAMING_URL": cache.get("STREAMING_URL")
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

    env_type = "PRODUCTION" if "api.ig.com" in api_url else "DEMO"
    last_error = None

    # Use reduced retries in startup mode to prevent blocking container initialization
    max_retries = LOGIN_MAX_RETRIES_STARTUP if startup_mode else LOGIN_MAX_RETRIES
    if startup_mode:
        print(f"🚀 IG login in startup mode ({env_type}) - reduced retries: {max_retries}")

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=LOGIN_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{api_url}/session",
                    headers=headers,
                    data=json.dumps(payload)
                )
                response.raise_for_status()

                cst = response.headers.get("CST")
                xst = response.headers.get("X-SECURITY-TOKEN")
                json_data = response.json()

                account_id = json_data.get("currentAccountId")
                streaming_url = json_data.get("lightstreamerEndpoint")

                print(f"✅ IG Login successful ({env_type}) - Account: {account_id}")

                # Store in URL-specific cache
                cache["CST"] = cst
                cache["X-SECURITY-TOKEN"] = xst
                cache["ACCOUNT_ID"] = account_id
                cache["STREAMING_URL"] = streaming_url
                cache["expires_at"] = time.time() + cache_ttl

                return {
                    "CST": cst,
                    "X-SECURITY-TOKEN": xst,
                    "ACCOUNT_ID": account_id,
                    "STREAMING_URL": streaming_url
                }

        except httpx.TimeoutException as e:
            last_error = e
            if attempt < max_retries - 1:
                backoff = LOGIN_RETRY_BACKOFF[attempt]
                print(f"⚠️ IG login timeout ({env_type}), retry {attempt + 1}/{max_retries} in {backoff}s...")
                await asyncio.sleep(backoff)
            else:
                print(f"❌ IG login failed after {max_retries} attempts ({env_type}): timeout")

        except httpx.HTTPStatusError as e:
            # Don't retry on 4xx errors (bad credentials, etc.)
            if 400 <= e.response.status_code < 500:
                print(f"❌ IG login failed ({env_type}): {e.response.status_code} - {e.response.text}")
                raise
            last_error = e
            if attempt < max_retries - 1:
                backoff = LOGIN_RETRY_BACKOFF[attempt]
                print(f"⚠️ IG login error ({env_type}): {e.response.status_code}, retry {attempt + 1}/{max_retries} in {backoff}s...")
                await asyncio.sleep(backoff)
            else:
                print(f"❌ IG login failed after {max_retries} attempts ({env_type}): {e}")

        except (httpx.ConnectError, httpx.ReadError) as e:
            last_error = e
            if attempt < max_retries - 1:
                backoff = LOGIN_RETRY_BACKOFF[attempt]
                print(f"⚠️ IG connection error ({env_type}), retry {attempt + 1}/{max_retries} in {backoff}s...")
                await asyncio.sleep(backoff)
            else:
                print(f"❌ IG login failed after {max_retries} attempts ({env_type}): connection error")

    # All retries exhausted
    raise last_error or Exception(f"IG login failed after {max_retries} attempts")
