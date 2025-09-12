# Ticker maps of IG specific ticker names
EPIC_MAP = {
    "EURUSD.1.MINI": "CS.D.EURUSD.MINI.IP",
    "GBPUSD.1.MINI": "CS.D.GBPUSD.MINI.IP",
    "USDJPY.100.MINI": "CS.D.USDJPY.MINI.IP",
    "AUDUSD.1.MINI": "CS.D.AUDUSD.MINI.IP"
    # Add more mappings as needed
}

API_BASE_URL = "https://api.ig.com/gateway/deal"  # or your demo/live base URL

# ================== AUTHENTICATION ==================
# Note: IG_API_KEY and IG_PWD values are now mapped to environment variables
# by the keyvault.py service. These names are kept for backward compatibility.
IG_USERNAME = "rehnmarkh"
IG_API_KEY = "prodapikey"
IG_PWD = "prodpwd"


