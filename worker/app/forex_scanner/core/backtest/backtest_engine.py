# core/backtest/backtest_engine.py
"""
Backtest Engine
Handles backtesting of trading strategies on historical data
FIXED: Timestamp sorting error
"""

import pandas as pd
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime, timezone, timedelta
import pytz

from ..strategies.ema_strategy import EMAStrategy
from ..strategies.macd_strategy import MACDStrategy
# from ..strategies.combined_strategy import CombinedStrategy  # Removed - strategy was disabled and unused
from ..strategies.bb_supertrend_strategy import BollingerSupertrendStrategy
from ..data_fetcher import DataFetcher
try:
    import config
except ImportError:
    from forex_scanner import config


class BacktestEngine:
    """Backtesting engine for strategy validation"""
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategies with BACKTEST MODE
        self.ema_strategy = EMAStrategy(backtest_mode=True)
        self.macd_strategy = MACDStrategy(backtest_mode=True)
        # self.combined_strategy = CombinedStrategy()  # Removed - strategy was disabled and unused
        
        # Initialize BB+Supertrend strategy if enabled
        if getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', False):
            try:
                bb_config = getattr(config, 'DEFAULT_BB_SUPERTREND_CONFIG', 'default')
                self.bb_supertrend_strategy = BollingerSupertrendStrategy(config_name=bb_config)
                self.logger.info(f"✅ BB+Supertrend Strategy initialized for backtesting with '{bb_config}' config")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize BB+Supertrend strategy for backtesting: {e}")
                self.bb_supertrend_strategy = None
        else:
            self.bb_supertrend_strategy = None
    
    def _convert_to_stockholm_time(self, utc_timestamp) -> datetime:
        """
        Convert UTC timestamp to Stockholm timezone (Europe/Stockholm)
        
        Args:
            utc_timestamp: UTC timestamp (various formats)
            
        Returns:
            datetime: Stockholm timezone datetime
        """
        try:
            # Define timezones
            utc_tz = pytz.UTC
            stockholm_tz = pytz.timezone('Europe/Stockholm')
            
            # Normalize the timestamp first
            if isinstance(utc_timestamp, str):
                dt = pd.to_datetime(utc_timestamp)
            elif isinstance(utc_timestamp, pd.Timestamp):
                dt = utc_timestamp.to_pydatetime()
            else:
                dt = utc_timestamp
            
            # Ensure it's timezone-aware UTC
            if dt.tzinfo is None:
                dt = utc_tz.localize(dt)
            elif dt.tzinfo != utc_tz:
                dt = dt.astimezone(utc_tz)
            
            # Convert to Stockholm time
            stockholm_dt = dt.astimezone(stockholm_tz)
            
            # Return as timezone-naive datetime (but in Stockholm time)
            return stockholm_dt.replace(tzinfo=None)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not convert timestamp to Stockholm time: {e}")
            # Fallback: add 2 hours (approximate for Stockholm)
            if isinstance(utc_timestamp, str):
                dt = pd.to_datetime(utc_timestamp)
            elif isinstance(utc_timestamp, pd.Timestamp):
                dt = utc_timestamp.to_pydatetime()
            else:
                dt = utc_timestamp
            
            return dt + timedelta(hours=2)
    
    def _normalize_timestamp(self, timestamp: Union[str, pd.Timestamp, datetime]) -> datetime:
        """
        FIXED: Normalize different timestamp types to timezone-naive datetime for consistent sorting
        
        Args:
            timestamp: Timestamp in various formats (str, pd.Timestamp, datetime)
            
        Returns:
            datetime: Normalized timezone-naive datetime object
        """
        try:
            if isinstance(timestamp, str):
                # Handle string timestamps
                dt = pd.to_datetime(timestamp)
                # Convert to timezone-naive datetime
                if dt.tz is not None:
                    dt = dt.tz_convert('UTC').tz_localize(None)
                return dt.to_pydatetime()
            elif isinstance(timestamp, pd.Timestamp):
                # Handle pandas Timestamp
                if timestamp.tz is not None:
                    # Convert timezone-aware to UTC then remove timezone
                    timestamp = timestamp.tz_convert('UTC').tz_localize(None)
                return timestamp.to_pydatetime()
            elif isinstance(timestamp, datetime):
                # Handle datetime objects
                if timestamp.tzinfo is not None:
                    # Convert timezone-aware to UTC then remove timezone
                    utc_dt = timestamp.astimezone(timezone.utc)
                    return utc_dt.replace(tzinfo=None)
                else:
                    # Already timezone-naive
                    return timestamp
            else:
                # Fallback: try converting to pandas timestamp first
                dt = pd.to_datetime(timestamp)
                if dt.tz is not None:
                    dt = dt.tz_convert('UTC').tz_localize(None)
                return dt.to_pydatetime()
        except Exception as e:
            # If all else fails, use current time as fallback (timezone-naive)
            self.logger.warning(f"⚠️ Could not normalize timestamp {timestamp}: {e}, using current time")
            return datetime.now().replace(tzinfo=None)
    
    def backtest_signals(
        self,
        epic_list: List[str],
        lookback_days: int = 30,
        use_bid_adjustment: bool = True,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> List[Dict]:
        """
        Backtest signals on historical data using all enabled strategies
        
        Args:
            epic_list: List of epics to test
            lookback_days: Days of historical data
            use_bid_adjustment: Whether to use BID adjustment
            spread_pips: Spread in pips
            timeframe: Timeframe to analyze
            
        Returns:
            List of historical signals with performance data
        """
        # Use default timeframe if not specified (CRITICAL FIX for alignment with live signals)
        if timeframe is None:
            timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '15m')
            self.logger.info(f"🔧 Using DEFAULT_TIMEFRAME from config: {timeframe}")
        
        self.logger.info(f"🔙 Starting backtest: {len(epic_list)} pairs, {lookback_days} days, {timeframe} timeframe")
        
        all_signals = []
        
        for epic in epic_list:
            try:
                pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
                pair = pair_info['pair']
                
                self.logger.info(f"Backtesting {epic} ({pair}) on {timeframe}...")

                # CRITICAL FIX: Use optimal lookback hours (same as live scanner)
                # This ensures backtest uses identical historical window
                optimal_lookback_hours = self._get_optimal_lookback_hours(
                    epic, timeframe, self.ema_strategy
                )

                # Log comparison for validation
                old_lookback = lookback_days * 24
                if optimal_lookback_hours != old_lookback:
                    self.logger.warning(
                        f"⚠️ Lookback mismatch for {epic}: "
                        f"old={old_lookback}h, optimal={optimal_lookback_hours}h"
                    )

                # Get historical data with optimal lookback
                df_data = self.data_fetcher.get_enhanced_data(
                    epic, pair, timeframe=timeframe,
                    lookback_hours=optimal_lookback_hours
                )
                
                if df_data is None or len(df_data) < config.MIN_BARS_FOR_SIGNAL:
                    self.logger.warning(f"Insufficient data for {epic}")
                    continue
                
                # Get historical signals from all enabled strategies
                signals = self._backtest_epic_all_strategies(
                    df=df_data, 
                    epic=epic, 
                    pair=pair,
                    use_bid_adjustment=use_bid_adjustment,
                    spread_pips=spread_pips, 
                    timeframe=timeframe
                )
                
                all_signals.extend(signals)
                self.logger.info(f"Found {len(signals)} signals for {epic}")
                
            except Exception as e:
                self.logger.error(f"Error backtesting {epic}: {e}")
                continue
        
        # Convert all timestamps to Stockholm timezone for user-friendly display
        self.logger.info("🕐 Converting timestamps to Stockholm timezone...")
        for signal in all_signals:
            if 'timestamp' in signal:
                try:
                    utc_timestamp = signal['timestamp']
                    stockholm_timestamp = self._convert_to_stockholm_time(utc_timestamp)
                    # Convert to string format to ensure proper display
                    signal['timestamp'] = stockholm_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    signal['timezone'] = 'Europe/Stockholm'  # Add timezone indicator
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not convert timestamp for signal: {e}")
        
        # FIXED: Sort by timestamp with proper type normalization
        try:
            all_signals.sort(key=lambda x: self._normalize_timestamp(x['timestamp']), reverse=True)
        except Exception as e:
            self.logger.error(f"❌ Error sorting signals by timestamp: {e}")
            # Fallback: try basic sorting or return unsorted
            try:
                # Try sorting without reverse first
                all_signals.sort(key=lambda x: str(x['timestamp']))
            except:
                self.logger.warning("⚠️ Could not sort signals by timestamp, returning unsorted")
        
        self.logger.info(f"📊 Backtest complete: {len(all_signals)} total signals")
        self.logger.info("✅ All timestamps displayed in Stockholm timezone (Europe/Stockholm)")
        return all_signals
    
    
    def _backtest_epic_all_strategies(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        pair: str,
        use_bid_adjustment: bool = True,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> List[Dict]:
        """Backtest single epic with all enabled strategies"""
        # Use default timeframe if not specified
        if timeframe is None:
            timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '15m')
        
        all_signals = []
        
        # Check combined strategy mode value, not just existence
        combined_strategy_mode = getattr(config, 'COMBINED_STRATEGY_MODE', None)
        
        # Combined mode is enabled only if:
        # 1. COMBINED_STRATEGY_MODE has a valid value (not None)
        # 2. The value is one of the valid modes
        if combined_strategy_mode in ['consensus', 'weighted']:
            self.logger.info(f"🎯 Using combined strategy mode: {combined_strategy_mode}")
            signals = self._backtest_combined_strategy(df, epic, spread_pips, timeframe)
            all_signals.extend(signals)
            if signals:
                self.logger.info(f"   Combined strategy found {len(signals)} signals")
        else:
            self.logger.info("🎯 Using individual strategies for backtest")
            
            # Individual strategies
            if getattr(config, 'SIMPLE_EMA_STRATEGY', False):
                self.logger.info("   Running EMA strategy backtest...")
                signals = self._backtest_ema_strategy(df, epic, spread_pips, timeframe)
                all_signals.extend(signals)
                if signals:
                    self.logger.info(f"   EMA strategy found {len(signals)} signals")
            
            if getattr(config, 'MACD_EMA_STRATEGY', False):
                self.logger.info("   Running MACD strategy backtest...")
                signals = self._backtest_macd_strategy(df, epic, spread_pips, timeframe)
                all_signals.extend(signals)
                if signals:
                    self.logger.info(f"   MACD strategy found {len(signals)} signals")
            
            # KAMA strategy
            if getattr(config, 'KAMA_STRATEGY', False):
                self.logger.info("   Running KAMA strategy backtest...")
                signals = self._backtest_kama_strategy(df, epic, spread_pips, timeframe)
                all_signals.extend(signals)
                if signals:
                    self.logger.info(f"   KAMA strategy found {len(signals)} signals")
            
            # BB+Supertrend strategy
            if getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', False) and self.bb_supertrend_strategy:
                self.logger.info("   Running BB+Supertrend strategy backtest...")
                signals = self._backtest_bb_supertrend_strategy(df, epic, spread_pips, timeframe)
                all_signals.extend(signals)
                if signals:
                    self.logger.info(f"   BB+Supertrend strategy found {len(signals)} signals")
        
        self.logger.info(f"📈 Strategy scan complete: {len(all_signals)} signals found")
        return all_signals
    
    def _backtest_ema_strategy(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[Dict]:
        """Backtest EMA strategy on historical data"""
        if not hasattr(self, 'ema_strategy') or self.ema_strategy is None:
            try:
                self.ema_strategy = EMAStrategy(backtest_mode=True)
                self.logger.info("✅ EMA strategy initialized for backtest")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize EMA strategy: {e}")
                return []
        
        return self._scan_historical_signals_with_strategy(
            df, epic, self.ema_strategy, spread_pips, timeframe
        )
    
    def _backtest_macd_strategy(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[Dict]:
        """Backtest MACD strategy on historical data"""
        if not hasattr(self, 'macd_strategy') or self.macd_strategy is None:
            try:
                self.macd_strategy = MACDStrategy(backtest_mode=True)
                self.logger.info("✅ MACD strategy initialized for backtest")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize MACD strategy: {e}")
                return []
        
        return self._scan_historical_signals_with_strategy(
            df, epic, self.macd_strategy, spread_pips, timeframe
        )
    
    def _backtest_combined_strategy(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[Dict]:
        """Backtest combined strategy on historical data"""
        if not hasattr(self, 'combined_strategy') or self.combined_strategy is None:
            try:
                # self.combined_strategy = CombinedStrategy()  # Removed - strategy was disabled and unused
                self.logger.info("✅ Combined strategy initialized for backtest")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize combined strategy: {e}")
                return []
        
        return self._scan_historical_signals_with_strategy(
            df, epic, self.combined_strategy, spread_pips, timeframe
        )
    
    def _backtest_bb_supertrend_strategy(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[Dict]:
        """Backtest BB+Supertrend strategy on historical data"""
        if not hasattr(self, 'bb_supertrend_strategy') or self.bb_supertrend_strategy is None:
            try:
                from ..strategies.bb_supertrend_strategy import BollingerSupertrendStrategy
                bb_config = getattr(config, 'DEFAULT_BB_SUPERTREND_CONFIG', 'default')
                self.bb_supertrend_strategy = BollingerSupertrendStrategy(config_name=bb_config)
                self.logger.info("✅ BB+Supertrend strategy initialized for backtest")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize BB+Supertrend strategy: {e}")
                return []
        
        return self._scan_historical_signals_with_strategy(
            df, epic, self.bb_supertrend_strategy, spread_pips, timeframe
        )
    
    def _backtest_kama_strategy(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[Dict]:
        """Backtest KAMA strategy on historical data"""
        if not hasattr(self, 'kama_strategy') or self.kama_strategy is None:
            # Initialize KAMA strategy if not already done
            try:
                from ..strategies.kama_strategy import KAMAStrategy
                self.kama_strategy = KAMAStrategy()
                self.logger.info("✅ KAMA strategy initialized for backtest")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize KAMA strategy: {e}")
                return []
        
        return self._scan_historical_signals_with_strategy(
            df, epic, self.kama_strategy, spread_pips, timeframe
        )

    def _scan_historical_signals_with_strategy(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        strategy,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> List[Dict]:
        """Scan DataFrame for historical signals using specific strategy"""
        # Use default timeframe if not specified  
        if timeframe is None:
            timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '15m')
        
        signals = []
        start_idx = config.MIN_BARS_FOR_SIGNAL  # Start after we have enough data

        self.logger.info(f"🔍 Scanning {len(df)} bars from index {start_idx} to {len(df)-1}")

        for i in range(start_idx + 1, len(df)):
            try:
                # CRITICAL FIX: Use FULL historical window (same as live scanner)
                # Live scanner gets ALL available historical data for each signal
                # Backtest MUST do the same for accurate comparison
                history_start = 0  # Use all available history
                current_data = df.iloc[history_start:i+1].copy()

                # Log first iteration to verify full window usage
                if i == start_idx + 1:
                    self.logger.info(
                        f"✅ Using FULL historical window: "
                        f"{len(current_data)} bars (same as live scanner)"
                    )
                
                if len(current_data) < 2:
                    continue
                
                # Detect signal using the strategy
                signal = strategy.detect_signal(current_data, epic, spread_pips, timeframe)
                
                if signal:
                    # Add performance metrics by looking ahead
                    signal = self._add_backtest_performance(signal, df, i)
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error scanning at index {i}: {e}")
                continue
        
        self.logger.info(f"📈 Strategy scan complete: {len(signals)} signals found")
        return signals
    
    def _add_backtest_performance(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """Add performance metrics to signal by looking ahead"""
        try:
            signal_enhanced = signal.copy()
            
            # Get signal details
            entry_price = signal_enhanced.get('entry_price', df.iloc[signal_idx]['close'])
            signal_type = signal_enhanced.get('signal_type', 'UNKNOWN')
            
            # Look ahead for performance (up to 24 hours / 96 bars for 15m timeframe)
            max_lookback = min(96, len(df) - signal_idx - 1)
            
            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]
                
                if signal_type == 'BULL':
                    # For buy signals, look for profit in higher prices
                    max_profit = (future_data['high'].max() - entry_price) * 10000  # Convert to pips
                    max_loss = (entry_price - future_data['low'].min()) * 10000
                elif signal_type == 'BEAR':
                    # For sell signals, look for profit in lower prices
                    max_profit = (entry_price - future_data['low'].min()) * 10000  # Convert to pips
                    max_loss = (future_data['high'].max() - entry_price) * 10000
                else:
                    max_profit = 0
                    max_loss = 0
                
                signal_enhanced.update({
                    'max_profit_pips': round(max_profit, 1),
                    'max_loss_pips': round(max_loss, 1),
                    'profit_loss_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
                    'lookback_bars': max_lookback
                })
            
            return signal_enhanced

        except Exception as e:
            self.logger.error(f"Error adding performance metrics: {e}")
            return signal

    def _get_optimal_lookback_hours(self, epic: str, timeframe: str, ema_strategy=None) -> int:
        """
        Get optimal lookback hours - SAME LOGIC AS LIVE SCANNER
        This ensures backtest uses identical historical window as live trading
        """
        try:
            # Base lookback hours by timeframe (MUST MATCH DataFetcher logic)
            base_lookback = {
                '1m': 24,    # 1 day for 1m (1440 bars)
                '5m': 48,    # 2 days for 5m (576 bars)
                '15m': 168,  # 1 week for 15m (672 bars)
                '1h': 720,   # 1 month for 1h (720 bars)
                '1d': 8760   # 1 year for 1d (365 bars)
            }.get(timeframe, 48)

            # Adjust for dynamic EMA requirements
            if ema_strategy and hasattr(ema_strategy, 'get_ema_periods'):
                try:
                    ema_periods = ema_strategy.get_ema_periods(epic)
                    max_ema = max(ema_periods) if ema_periods else 200

                    # Ensure we have enough data for the largest EMA (at least 3x the period)
                    min_bars_needed = max_ema * 3
                    timeframe_minutes = self._timeframe_to_minutes(timeframe)
                    min_hours_needed = (min_bars_needed * timeframe_minutes) / 60

                    base_lookback = max(base_lookback, int(min_hours_needed))
                    self.logger.debug(f"🔧 Adjusted lookback for EMA {max_ema}: {base_lookback}h")
                except Exception as e:
                    self.logger.debug(f"Could not get EMA periods: {e}")

            # Adjust for KAMA requirements if enabled
            if getattr(config, 'KAMA_STRATEGY', False):
                kama_configs = getattr(config, 'KAMA_STRATEGY_CONFIG', {})
                default_config = getattr(config, 'DEFAULT_KAMA_CONFIG', 'default')

                if default_config in kama_configs:
                    kama_period = kama_configs[default_config].get('period', 10)
                    min_bars_needed = kama_period * 3
                    timeframe_minutes = self._timeframe_to_minutes(timeframe)
                    min_hours_needed = (min_bars_needed * timeframe_minutes) / 60

                    base_lookback = max(base_lookback, int(min_hours_needed))

            self.logger.info(f"📊 Optimal lookback for {timeframe}: {base_lookback}h")
            return base_lookback

        except Exception as e:
            self.logger.error(f"Error calculating optimal lookback: {e}")
            # Fallback to conservative default
            return 168  # 1 week

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            return 15  # Default to 15 minutes