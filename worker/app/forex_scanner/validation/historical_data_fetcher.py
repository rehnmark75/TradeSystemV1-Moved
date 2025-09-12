# validation/historical_data_fetcher.py
"""
Historical Data Fetcher - Mock DataFetcher for Validation

This module provides a DataFetcher-compatible interface that returns
pre-prepared historical data instead of fetching live data. This ensures
the validation system uses the same strategies and detection logic as
the live system but with historical data.
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from forex_scanner.core.data_fetcher import DataFetcher


class HistoricalDataFetcher:
    """
    DataFetcher-compatible class that returns pre-prepared historical data
    
    This class implements the same interface as DataFetcher but returns
    historical data that has been prepared by HistoricalDataManager.
    This allows the live SignalDetector and strategies to work unchanged
    with historical data for validation purposes.
    """
    
    def __init__(self, prepared_data: pd.DataFrame, user_timezone: str = 'Europe/Stockholm'):
        """
        Initialize with pre-prepared historical data
        
        Args:
            prepared_data: DataFrame with historical data and calculated indicators
            user_timezone: Timezone for compatibility
        """
        self.prepared_data = prepared_data
        self.user_timezone = user_timezone
        self.logger = logging.getLogger(__name__)
        
        # For compatibility with DataFetcher interface
        self.db_manager = None
        
        self.logger.info(f"🕒 HistoricalDataFetcher initialized with {len(prepared_data)} bars")
        if 'start_time' in prepared_data.columns:
            start_time = prepared_data['start_time'].min()
            end_time = prepared_data['start_time'].max()
            self.logger.info(f"   Time range: {start_time} to {end_time}")
    
    def get_enhanced_data(
        self, 
        epic: str, 
        pair: str, 
        timeframe: str = '15m',
        ema_strategy=None,
        macd_strategy=None,
        kama_strategy=None,
        zero_lag_strategy=None
    ) -> Optional[pd.DataFrame]:
        """
        Return the pre-prepared historical data
        
        This method signature matches DataFetcher.get_enhanced_data()
        but simply returns the pre-prepared data instead of fetching live data.
        
        Args:
            epic: Epic code (ignored - data already prepared for specific epic)
            pair: Trading pair (ignored)
            timeframe: Timeframe (ignored)
            ema_strategy: EMA strategy (ignored)
            macd_strategy: MACD strategy (ignored)
            kama_strategy: KAMA strategy (ignored)
            zero_lag_strategy: Zero lag strategy (ignored)
            
        Returns:
            Pre-prepared DataFrame with all indicators calculated
        """
        self.logger.debug(f"📊 Returning pre-prepared historical data for {epic}")
        self.logger.debug(f"   Data shape: {self.prepared_data.shape}")
        
        # Log the EMA columns to verify they're present
        ema_cols = [col for col in self.prepared_data.columns if col.startswith('ema_')]
        self.logger.debug(f"   EMA columns: {ema_cols}")
        
        return self.prepared_data.copy()
    
    def get_candles_data(
        self, 
        epic: str, 
        pair: str, 
        timeframe: str = '15m', 
        lookback_hours: int = 72
    ) -> Optional[pd.DataFrame]:
        """Return basic candle data (OHLC) from prepared data"""
        base_cols = ['start_time', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in base_cols if col in self.prepared_data.columns]
        
        if not available_cols:
            self.logger.warning("⚠️ No basic candle data columns found")
            return None
            
        return self.prepared_data[available_cols].copy()
    
    def get_optimal_lookback_hours(self, epic: str, timeframe: str = '15m') -> int:
        """Return a reasonable lookback hours value"""
        return 72  # Default value for compatibility
    
    # Add other DataFetcher methods for compatibility
    def clear_ema_config_log_cache(self):
        """Compatibility method - does nothing"""
        pass
    
    def get_ema_config_log_stats(self):
        """Compatibility method"""
        return {'cached_configs': 0, 'memory_usage_estimate': 0, 'sample_entries': []}