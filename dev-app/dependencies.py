from services.ig_auth import ig_login
from services.keyvault import get_secret
from datetime import datetime, timedelta
from config import IG_USERNAME, IG_PWD, IG_API_KEY

# Production credentials (for VSL streaming - must match stream-app)
PROD_IG_USERNAME = "rehnmarkh"
PROD_IG_API_KEY = "prodapikey"
PROD_IG_PWD = "prodpwd"
PROD_API_BASE_URL = "https://api.ig.com/gateway/deal"

# Global cache (in-memory or move to Redis if production)
ig_token_cache = {
    "CST": None,
    "X-SECURITY-TOKEN": None,
    "stream_used": False,
    "expires_at": None
}


# Global in-memory token cache
ig_token_cache = {
    "CST": None,
    "X-SECURITY-TOKEN": None,
    "ACCOUNT_ID": None,
    "stream_used": False,
    "expires_at": None
}

async def get_ig_auth_headers(force_refresh: bool = False):
    now = datetime.utcnow()

    # Check if refresh is needed
    should_refresh = (
        force_refresh or
        ig_token_cache["CST"] is None or
        ig_token_cache["X-SECURITY-TOKEN"] is None or
        ig_token_cache["ACCOUNT_ID"] is None or
        ig_token_cache["expires_at"] is None or
        now >= ig_token_cache["expires_at"] or
        ig_token_cache["stream_used"]
    )

    if should_refresh:
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

    return {
        "Accept": "application/json; charset=UTF-8",
        "Content-Type": "application/json; charset=UTF-8",
        "X-IG-API-KEY": get_secret(IG_API_KEY),  # Reuse securely
        "Version": "2",
        "CST": ig_token_cache["CST"],
        "X-SECURITY-TOKEN": ig_token_cache["X-SECURITY-TOKEN"],
        "accountId": ig_token_cache["ACCOUNT_ID"]
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
    """
    now = datetime.utcnow()

    # Check if refresh is needed
    should_refresh = (
        force_refresh or
        prod_token_cache["CST"] is None or
        prod_token_cache["X-SECURITY-TOKEN"] is None or
        prod_token_cache["ACCOUNT_ID"] is None or
        prod_token_cache["expires_at"] is None or
        now >= prod_token_cache["expires_at"]
    )

    if should_refresh:
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

    return {
        "Accept": "application/json; charset=UTF-8",
        "Content-Type": "application/json; charset=UTF-8",
        "X-IG-API-KEY": get_secret(PROD_IG_API_KEY),
        "Version": "2",
        "CST": prod_token_cache["CST"],
        "X-SECURITY-TOKEN": prod_token_cache["X-SECURITY-TOKEN"],
        "accountId": prod_token_cache["ACCOUNT_ID"]
    }
