import asyncio
from services.ig_auth import ig_login
from services.keyvault import get_secret
from datetime import datetime, timedelta
from config import IG_USERNAME, IG_PWD, IG_API_KEY

# Production credentials (for VSL streaming - must match stream-app)
PROD_IG_USERNAME = "rehnmarkh"
PROD_IG_API_KEY = "prodapikey"
PROD_IG_PWD = "prodpwd"
PROD_API_BASE_URL = "https://api.ig.com/gateway/deal"

# Locks to prevent concurrent token refresh race conditions
# Without these, multiple async tasks could all see expired token and call ig_login simultaneously
# NOTE: Locks must be created lazily to avoid binding to old event loops after restart
_demo_token_lock = None
_prod_token_lock = None

def _get_demo_lock():
    """Get or create demo token lock in current event loop."""
    global _demo_token_lock
    try:
        # Check if lock exists and is valid for current event loop
        if _demo_token_lock is not None:
            # Try to access the loop - will raise if wrong event loop
            _demo_token_lock._loop
            return _demo_token_lock
    except (AttributeError, RuntimeError):
        pass

    # Create new lock in current event loop
    _demo_token_lock = asyncio.Lock()
    return _demo_token_lock

def _get_prod_lock():
    """Get or create prod token lock in current event loop."""
    global _prod_token_lock
    try:
        # Check if lock exists and is valid for current event loop
        if _prod_token_lock is not None:
            # Try to access the loop - will raise if wrong event loop
            _prod_token_lock._loop
            return _prod_token_lock
    except (AttributeError, RuntimeError):
        pass

    # Create new lock in current event loop
    _prod_token_lock = asyncio.Lock()
    return _prod_token_lock

# Global in-memory token cache (demo account)
ig_token_cache = {
    "CST": None,
    "X-SECURITY-TOKEN": None,
    "ACCOUNT_ID": None,
    "stream_used": False,
    "expires_at": None
}

async def get_ig_auth_headers(force_refresh: bool = False):
    """
    Get IG auth headers for demo account, with lock to prevent concurrent refresh.

    The lock ensures that if multiple async tasks need to refresh the token simultaneously,
    only one actually calls ig_login() while others wait and then use the cached result.
    """
    now = datetime.utcnow()

    # Quick check without lock - if token is valid, return immediately
    if not force_refresh and _is_token_valid(ig_token_cache, now):
        return _build_headers(ig_token_cache, IG_API_KEY)

    # Token needs refresh - acquire lock to prevent concurrent refresh
    async with _get_demo_lock():
        # Double-check after acquiring lock (another task may have refreshed)
        now = datetime.utcnow()
        if not force_refresh and _is_token_valid(ig_token_cache, now):
            return _build_headers(ig_token_cache, IG_API_KEY)

        print("🔄 Refreshing IG token...")

        api_key = get_secret(IG_API_KEY)
        ig_pwd = get_secret(IG_PWD)
        ig_usr = IG_USERNAME

        tokens = await ig_login(api_key=api_key, ig_pwd=ig_pwd, ig_usr=ig_usr)

        ig_token_cache["CST"] = tokens["CST"]
        ig_token_cache["X-SECURITY-TOKEN"] = tokens["X-SECURITY-TOKEN"]
        ig_token_cache["ACCOUNT_ID"] = tokens["ACCOUNT_ID"]
        ig_token_cache["stream_used"] = False
        ig_token_cache["expires_at"] = now + timedelta(minutes=55)  # Leave buffer

    return _build_headers(ig_token_cache, IG_API_KEY)


def _is_token_valid(cache: dict, now: datetime) -> bool:
    """Check if cached token is still valid."""
    return (
        cache["CST"] is not None and
        cache["X-SECURITY-TOKEN"] is not None and
        cache["ACCOUNT_ID"] is not None and
        cache["expires_at"] is not None and
        now < cache["expires_at"] and
        not cache.get("stream_used", False)
    )


def _build_headers(cache: dict, api_key_name: str) -> dict:
    """Build headers dict from cache."""
    return {
        "Accept": "application/json; charset=UTF-8",
        "Content-Type": "application/json; charset=UTF-8",
        "X-IG-API-KEY": get_secret(api_key_name),
        "Version": "2",
        "CST": cache["CST"],
        "X-SECURITY-TOKEN": cache["X-SECURITY-TOKEN"],
        "accountId": cache["ACCOUNT_ID"]
    }


# Separate cache for PRODUCTION credentials (used by VSL streaming)
prod_token_cache = {
    "CST": None,
    "X-SECURITY-TOKEN": None,
    "ACCOUNT_ID": None,
    "expires_at": None
}


async def get_prod_auth_headers(force_refresh: bool = False):
    """
    Get PRODUCTION IG auth headers for VSL streaming.

    This uses production credentials (same as stream-app) because
    Lightstreamer streaming requires production account.

    Uses lock to prevent concurrent refresh race conditions.
    """
    now = datetime.utcnow()

    # Quick check without lock - if token is valid, return immediately
    if not force_refresh and _is_prod_token_valid(prod_token_cache, now):
        return _build_headers(prod_token_cache, PROD_IG_API_KEY)

    # Token needs refresh - acquire lock to prevent concurrent refresh
    async with _get_prod_lock():
        # Double-check after acquiring lock (another task may have refreshed)
        now = datetime.utcnow()
        if not force_refresh and _is_prod_token_valid(prod_token_cache, now):
            return _build_headers(prod_token_cache, PROD_IG_API_KEY)

        print("🔄 Refreshing PRODUCTION IG token for VSL...")

        api_key = get_secret(PROD_IG_API_KEY)
        ig_pwd = get_secret(PROD_IG_PWD)
        ig_usr = PROD_IG_USERNAME

        tokens = await ig_login(
            api_key=api_key,
            ig_pwd=ig_pwd,
            ig_usr=ig_usr,
            api_url=PROD_API_BASE_URL
        )

        prod_token_cache["CST"] = tokens["CST"]
        prod_token_cache["X-SECURITY-TOKEN"] = tokens["X-SECURITY-TOKEN"]
        prod_token_cache["ACCOUNT_ID"] = tokens["ACCOUNT_ID"]
        prod_token_cache["expires_at"] = now + timedelta(minutes=55)

        print(f"✅ PRODUCTION auth successful - Account: {tokens['ACCOUNT_ID']}")

    return _build_headers(prod_token_cache, PROD_IG_API_KEY)


def _is_prod_token_valid(cache: dict, now: datetime) -> bool:
    """Check if cached production token is still valid (no stream_used check)."""
    return (
        cache["CST"] is not None and
        cache["X-SECURITY-TOKEN"] is not None and
        cache["ACCOUNT_ID"] is not None and
        cache["expires_at"] is not None and
        now < cache["expires_at"]
    )
