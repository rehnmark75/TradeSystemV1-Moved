"""
Internal API authentication dependency for FastAPI.

Protects write endpoints from unauthorized access using a shared secret token.
The token is compared using hmac.compare_digest to prevent timing attacks.
"""

import hmac
import os
import logging

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)


def require_internal_token(x_internal_token: str = Header(None)):
    """
    FastAPI dependency that validates the X-Internal-Token header.

    Raises:
        503 if INTERNAL_API_TOKEN env var is not configured (service misconfigured)
        401 if the token is missing or does not match
    """
    expected = os.getenv('INTERNAL_API_TOKEN')
    if not expected:
        logger.error("INTERNAL_API_TOKEN not configured — rejecting request")
        raise HTTPException(
            status_code=503,
            detail="INTERNAL_API_TOKEN not configured — service disabled"
        )
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="Invalid internal token")
