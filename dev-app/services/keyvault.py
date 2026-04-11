import os
from functools import lru_cache

# Mapping from Azure KeyVault secret names to environment variable names
# In live mode, both demo* and prod* secret names resolve to IG_API_KEY/IG_PWD
# (which docker-compose sets to the PROD credentials)
_is_live = os.getenv('TRADING_ENVIRONMENT', 'demo') == 'live'

SECRET_ENV_MAP = {
    # Demo/Development credentials
    "demoapikey": "IG_API_KEY",  # Uses DEMO_API_KEY (demo) or PROD_API_KEY (live) from docker-compose
    "demopwd": "IG_PWD",         # Uses DEMO_PASSWORD (demo) or PROD_PASSWORD (live) from docker-compose
    # Production credentials
    "prodapikey": "IG_API_KEY" if _is_live else "PROD_IG_API_KEY",
    "prodpwd": "IG_PWD" if _is_live else "PROD_IG_PWD",
    # Other secrets
    "gemini": "GEMINI_API_KEY"
}

@lru_cache()
def get_secret(secret_name: str, keyvault_name: str = "tradersdata") -> str:
    """
    Get secret from environment variables instead of Azure KeyVault.
    
    Args:
        secret_name: The original KeyVault secret name
        keyvault_name: Ignored (kept for backward compatibility)
    
    Returns:
        Secret value from environment variable
        
    Raises:
        RuntimeError: If secret is not found in environment variables
    """
    try:
        # Map KeyVault secret name to environment variable name
        env_var_name = SECRET_ENV_MAP.get(secret_name)
        
        if not env_var_name:
            raise RuntimeError(f"Unknown secret name: {secret_name}")
        
        # Get value from environment variable
        secret_value = os.getenv(env_var_name)
        
        if secret_value is None:
            raise RuntimeError(f"Environment variable {env_var_name} not set for secret {secret_name}")
        
        if not secret_value.strip():
            raise RuntimeError(f"Environment variable {env_var_name} is empty for secret {secret_name}")
            
        return secret_value
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch secret '{secret_name}': {e}")

