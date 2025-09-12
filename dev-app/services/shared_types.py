# services/shared_types.py
"""
Shared types and configurations to avoid circular imports
This module contains common classes used by both trade_monitor and pair_specific_config
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import threading
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


@dataclass
class ApiConfig:
    """Configuration for API calls"""
    base_url: str = "http://fastapi-dev:8000"  # TODO: Use FASTAPI_DEV_URL from config
    subscription_key: str = "436abe054a074894a0517e5172f0e5b6"
    dry_run: bool = False  # Set to False for live trading
    
    @property
    def adjust_stop_url(self) -> str:
        """Get the full URL for the adjust-stop endpoint"""
        return f"{self.base_url}/orders/adjust-stop"


class PositionCache:
    """Efficient position caching to minimize IG API calls"""
    
    def __init__(self, cache_duration_seconds: int = 30):
        self.cache_duration = cache_duration_seconds
        self._positions_cache = None
        self._last_fetch_time = None
        self._lock = threading.Lock()
        
    def is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._last_fetch_time or not self._positions_cache:
            return False
        
        return (datetime.now() - self._last_fetch_time).total_seconds() < self.cache_duration
    
    async def get_positions(self, trading_headers: dict, force_refresh: bool = False) -> Optional[List[Dict]]:
        """Get positions with caching to minimize API calls"""
        with self._lock:
            if not force_refresh and self.is_cache_valid():
                return self._positions_cache
        
        try:
            import httpx
            from config import API_BASE_URL
            
            url = f"{API_BASE_URL}/positions"
            headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "2"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                positions = data.get("positions", [])
                
                # Atomic cache update
                with self._lock:
                    self._positions_cache = positions
                    self._last_fetch_time = datetime.now()
                
                return positions
                
        except Exception as e:
            logging.getLogger("trade_monitor").error(f"[POSITION CACHE ERROR] {e}")
            # Return stale cache if available, but don't update timestamp
            with self._lock:
                return self._positions_cache


class TradeMonitorLogger:
    """Centralized logging setup"""

    def __init__(self, log_file: str = "/app/logs/trade_monitor.log"):
        self.logger = logging.getLogger("trade_monitor")
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger