# optimization/data_collectors/base_collector.py
"""
Base collector class for unified parameter optimizer.
Provides common database access and data normalization.
"""

import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

try:
    from core.database import DatabaseManager
    import config
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner import config


class BaseCollector(ABC):
    """Base class for data collectors"""

    def __init__(self, db_manager: DatabaseManager = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        if db_manager is None:
            db_manager = DatabaseManager(config.DATABASE_URL)
        self.db_manager = db_manager

    @abstractmethod
    def collect(self, days: int = 30, epics: List[str] = None) -> pd.DataFrame:
        """
        Collect data from the source.

        Args:
            days: Number of days of historical data to collect
            epics: Optional list of epics to filter (None = all)

        Returns:
            DataFrame with collected data
        """
        pass

    def _execute_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        try:
            result = self.db_manager.execute_query(query, params or {})
            if result is None or (hasattr(result, 'empty') and result.empty):
                return pd.DataFrame()
            return result
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return pd.DataFrame()

    def _get_date_filter(self, days: int) -> datetime:
        """Get the start date for filtering"""
        return datetime.utcnow() - timedelta(days=days)

    def _normalize_direction(self, direction: str) -> str:
        """Normalize direction to BULL/BEAR"""
        if not direction:
            return None
        direction = str(direction).upper()
        if direction in ['BUY', 'LONG', 'BULL', 'UP']:
            return 'BULL'
        elif direction in ['SELL', 'SHORT', 'BEAR', 'DOWN']:
            return 'BEAR'
        return direction

    def _normalize_epic(self, epic: str) -> str:
        """Extract short pair name from epic"""
        if not epic:
            return None
        # CS.D.EURUSD.CEEM.IP -> EURUSD
        parts = epic.split('.')
        if len(parts) >= 3:
            return parts[2]
        return epic
