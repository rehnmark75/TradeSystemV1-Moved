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

# ================== POSITION CLOSER CONFIGURATION ==================
# Weekend protection - automatically close positions on Fridays
ENABLE_POSITION_CLOSER = True         # Enable/disable automatic position closure
POSITION_CLOSURE_HOUR_UTC = 20        # Hour to close positions (UTC)
POSITION_CLOSURE_MINUTE_UTC = 30      # Minute to close positions (UTC)
POSITION_CLOSURE_WEEKDAY = 4          # Day of week to close positions (4 = Friday, 0 = Monday)

# Position closer safety settings
POSITION_CLOSER_TIMEOUT_SECONDS = 60  # Timeout for position closure operations
POSITION_CLOSER_MAX_RETRY_ATTEMPTS = 3  # Max retries for failed closures
POSITION_CLOSER_RETRY_DELAY_SECONDS = 5  # Delay between retry attempts


