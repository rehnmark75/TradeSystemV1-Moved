"""
Import shim: exposes PAIR_TRAILING_CONFIGS and SCALP_TRAILING_CONFIGS from
the live fastapi-dev config, mounted read-only into task-worker as
/app/trailing_config_live.py via docker-compose.yml.

This is the single source of truth for trailing configs across live and backtest.
Never edit trailing values here — edit dev-app/config.py instead.
"""
import logging

try:
    from trailing_config_live import PAIR_TRAILING_CONFIGS, SCALP_TRAILING_CONFIGS  # noqa: F401
except ImportError:
    logging.getLogger(__name__).warning(
        "trailing_config_live.py not found — dev-app/config.py may not be mounted. "
        "Run: docker compose up -d --no-deps task-worker to pick up the new volume mount. "
        "Backtest trailing will use empty fallback configs."
    )
    PAIR_TRAILING_CONFIGS: dict = {}
    SCALP_TRAILING_CONFIGS: dict = {}
