from services.ig_auth import ig_login
from services.keyvault import get_secret
from config import IG_USERNAME, IG_API_KEY, IG_PWD

async def get_ig_auth_headers():
    api_key = get_secret(IG_API_KEY)
    ig_pwd = get_secret(IG_PWD)
    ig_usr = IG_USERNAME

    tokens = await ig_login(
        api_key=api_key,
        ig_pwd=ig_pwd,
        ig_usr=ig_usr
    )

    return {
        "Accept": "application/json; charset=UTF-8",
        "Content-Type": "application/json; charset=UTF-8",
        "X-IG-API-KEY": api_key,
        "Version": "2",  # ✅ must be "2" for secure session APIs
        "CST": tokens["CST"],
        "X-SECURITY-TOKEN": tokens["X-SECURITY-TOKEN"]
    }