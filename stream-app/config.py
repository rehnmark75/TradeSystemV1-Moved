# ================== IG AUTHENTICATION ==================
# Note: IG_API_KEY and IG_PWD values are now mapped to environment variables
# by the keyvault.py service. These names are kept for backward compatibility.
IG_USERNAME = "rehnmarkh"
IG_API_KEY = "prodapikey"
IG_PWD = "prodpwd"
IG_ACCOUNT_TYPE="LIVE"

# PG info
DATABASE_URL = "postgresql://postgres:postgres@postgres:5432/forex"

# Stream vs API validation settings
ENABLE_STREAM_API_VALIDATION = True
STREAM_VALIDATION_DELAY_SECONDS = 1800  # 30 minutes - allow IG API time to make historical data available 
STREAM_VALIDATION_FREQUENCY = 3
ENABLE_AUTOMATIC_PRICE_CORRECTION = True

STREAM_VALIDATION_THRESHOLDS = {
    'MINOR': 1.0,      # 1 pip - acceptable variance
    'MODERATE': 3.0,   # 3 pips - worth noting  
    'MAJOR': 10.0,     # 10 pips - significant issue
    'CRITICAL': 25.0   # 25+ pips - critical data integrity problem
}

# URL Configuration
# IG Markets API URLs
IG_API_BASE_URL = "https://api.ig.com/gateway/deal"
IG_DEMO_API_BASE_URL = "https://demo-api.ig.com/gateway/deal"

# Lightstreamer URLs  
LIGHTSTREAMER_PROD_URL = "https://apd.marketdatasystems.com"
LIGHTSTREAMER_DEMO_URL = "https://demo-apd.marketdatasystems.com"

# Azure URLs - REMOVED: Migrated from Azure KeyVault to environment variables

# Epic Configuration
# Active trading pairs for streaming and analysis
ACTIVE_EPICS = [
    "CS.D.GBPUSD.MINI.IP",
    "CS.D.USDJPY.MINI.IP", 
    "CS.D.AUDUSD.MINI.IP",
    "CS.D.USDCAD.MINI.IP",
    "CS.D.EURJPY.MINI.IP",
    "CS.D.AUDJPY.MINI.IP",
    "CS.D.NZDUSD.MINI.IP",
    "CS.D.USDCHF.MINI.IP",
    "CS.D.EURUSD.CEEM.IP"
]

# Additional available epics (currently disabled)
ADDITIONAL_EPICS = [
    "CS.D.CADJPY.MINI.IP",
    "CS.D.CHFJPY.MINI.IP",
    "CS.D.GBPJPY.MINI.IP", 
    "CS.D.NZDJPY.MINI.IP"
]

# All available epics (for validation/testing)
ALL_EPICS = ACTIVE_EPICS + ADDITIONAL_EPICS

# Specific epic for testing/validation (commonly used in validation scripts)
DEFAULT_TEST_EPIC = "CS.D.EURUSD.CEEM.IP"