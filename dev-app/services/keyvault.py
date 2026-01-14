import os
from functools import lru_cache

# Mapping from Azure KeyVault secret names to environment variable names
SECRET_ENV_MAP = {
    # Demo/Development credentials for dev-app
    "demoapikey": "IG_API_KEY",  # Uses DEMO_API_KEY from docker-compose
    "demopwd": "IG_PWD",         # Uses DEMO_PASSWORD from docker-compose
    # Production credentials (for VSL streaming - must use production account)
    "prodapikey": "PROD_IG_API_KEY",  # Uses PROD_API_KEY from docker-compose
    "prodpwd": "PROD_IG_PWD",         # Uses PROD_PASSWORD from docker-compose
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

